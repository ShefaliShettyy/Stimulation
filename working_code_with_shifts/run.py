import copy
import time

from core_simulation import (
    BehaviorConfig,
    SimulationConfig,
    SimulationEngine,
)
from cost_system import (
    CostAwareEngine,
    CostAwareRealisticsEngine,
    CostConfig,
    CostReport,
    ScenarioCostResult,
)
from optimization import (
    COVERAGE_BANDS,
    HORIZON_MINUTES,
    PEAK_BAND_INDEX,
    SHIFT_WINDOWS,
    _SHIFT_BREAK_MIDPOINTS,
    _BREAK_DURATION_MINUTES,
    ErlangC,
    OptimizationConfig,
    OptimizationResult,
    OptimizeSimulateLoop,
    StaffingOptimizer,
)
from ml_system import (
    FeedbackConfig,
    FeedbackLoopReport,
    FeedbackLoopRunner,
    MLModelRegistry,
    MLSimulationEngine,
)


# ============================================================================
# HELPERS
# ============================================================================

def section(title: str, width: int = 72) -> None:
    bar = "═" * width
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def divider(width: int = 72) -> None:
    print("─" * width)


def print_shift_plan(opt_result: OptimizationResult, wages: dict, overhead: float) -> None:
    """Pretty-print the full multi-shift plan with per-band coverage table."""
    SHIFT_HOURS = 8.0
    print(f"\n  ┌─ Shift Plan  ({HORIZON_MINUTES}-min horizon: 3 overlapping 8-hr shifts) ─────┐")
    for sh_name, sh_start, sh_end in SHIFT_WINDOWS:
        plan       = opt_result.shift_plan.get(sh_name, {})
        shift_cost = sum(plan.get(sk, 0) * wages.get(sk, 18.0) * overhead * SHIFT_HOURS for sk in plan)
        agents_str = "  ".join(f"{sk}: {n}" for sk, n in sorted(plan.items()))
        break_min  = _SHIFT_BREAK_MIDPOINTS[sh_name]
        print(f"  │  Shift {sh_name}  ({sh_start:>4}–{sh_end:>4} min)  break@{break_min:.0f}m"
              f"  [{agents_str}]  cost=£{shift_cost:,.2f}")
    print(f"  └" + "─" * 68)

    skills    = sorted(opt_result.agents_per_skill.keys())
    hdr_skills = "".join(f"  {s[:8]:>9}" for s in skills)
    print(f"\n  {'Band interval':<22}  {'Shifts':>9}{hdr_skills}")
    divider(22 + 12 + 10 * len(skills))
    for band_idx, (b_start, b_end, active_shifts) in enumerate(COVERAGE_BANDS):
        cov      = opt_result.band_coverage.get(band_idx, {})
        sh_str   = "+".join(active_shifts)
        agents_s = "".join(f"  {cov.get(sk, 0):>9}" for sk in skills)
        peak_tag = "  ← PEAK" if band_idx == PEAK_BAND_INDEX else ""
        print(f"  [{b_start:>4}–{b_end:>4} min]        {sh_str:>9}{agents_s}{peak_tag}")


def build_shift_aware_break_schedule(opt_result: OptimizationResult) -> list:
    """
    FIX-9: Build a break schedule that reflects actual shift assignments.

    The SimulationEngine assigns the same break_schedule to every agent
    in the pool via _schedule_breaks().  The pool is sized to peak-band
    headcount (all three shifts' agents active simultaneously at the peak).

    Strategy: distribute peak-band agents across the three shifts by
    cycling M→D→E per skill.  For each agent slot, record which shift
    it belongs to, then assign that shift's break midpoint.

    The resulting break_schedule is a list of (start_minute, duration)
    tuples, one per agent in pool order.  _schedule_breaks() iterates
    agents and break_schedule in parallel — this is a LIST OF LISTS
    pattern, but since SimulationConfig.break_schedule is a flat list
    of (start, duration) entries applied identically to all agents, we
    instead return the MOST REPRESENTATIVE single break schedule:
    one entry per shift at the correct midpoint, 30 min each.

    This is the correct fix for the pool-level break schedule: keep 3
    entries (one per shift), which means every agent sees all three
    entries but only the one matching their actual shift midpoint is
    'in their future' — the others either fire as ~0-delay no-ops
    (if start < now) or fire well outside their work window (acceptable
    since they are already off-shift).

    Shift break midpoints:
      M (0–480)   → 240
      D (120–600) → 360
      E (240–720) → 480
    """
    return [
        (_SHIFT_BREAK_MIDPOINTS["M"], _BREAK_DURATION_MINUTES),
        (_SHIFT_BREAK_MIDPOINTS["D"], _BREAK_DURATION_MINUTES),
        (_SHIFT_BREAK_MIDPOINTS["E"], _BREAK_DURATION_MINUTES),
    ]


def build_optimised_sim_cfg(
    base_cfg:   SimulationConfig,
    opt_result: OptimizationResult,
) -> SimulationConfig:
    """
    FIX-9: Build the SimulationConfig for Stage 4 with:
      - peak-band agents_per_skill
      - 720-min horizon
      - shift-aware break schedule (3 entries, one per shift midpoint)
    """
    cfg = copy.copy(base_cfg)
    cfg.agents_per_skill     = dict(opt_result.agents_per_skill)
    cfg.sim_duration_minutes = float(HORIZON_MINUTES)
    cfg.break_schedule       = build_shift_aware_break_schedule(opt_result)
    return cfg


# ============================================================================
# CONFIGURATION
# ============================================================================

SEED                   = 42
SIM_DURATION_MINUTES   = float(HORIZON_MINUTES)   # 720
ARRIVAL_RATE_PER_HOUR  = 60.0
SLA_TARGET             = 0.90
SLA_THRESHOLD_MINUTES  = 1.0
MAX_AGENTS_PER_SKILL   = 15
MIN_AGENTS_PER_SHIFT   = 1        # FIX-E: minimum 1 per shift; band Erlang-C
                                  # constraints are the real SLA floor.  min=2
                                  # forced symmetric 2-2-2 plans and over-staffed
                                  # peak from 3 agents (Erlang-C) to 6 (3×min=2).
MAX_OPT_ITERATIONS     = 8
CONVERGENCE_TOLERANCE  = 0.015
ML_EPOCHS              = 5

sim_cfg = SimulationConfig(
    sim_duration_minutes  = SIM_DURATION_MINUTES,
    arrival_rate_per_hour = ARRIVAL_RATE_PER_HOUR,
    random_seed           = SEED,
    # FIX-10: 720-min break schedule for default pool too
    break_schedule        = [
        (_SHIFT_BREAK_MIDPOINTS["M"], _BREAK_DURATION_MINUTES),
        (_SHIFT_BREAK_MIDPOINTS["D"], _BREAK_DURATION_MINUTES),
        (_SHIFT_BREAK_MIDPOINTS["E"], _BREAK_DURATION_MINUTES),
    ],
)
cost_cfg = CostConfig()
beh_cfg  = BehaviorConfig()


# ============================================================================
# STAGE 0 — Erlang-C analytical preview
# ============================================================================

def stage0_erlang_preview() -> None:
    section("STAGE 0 — Erlang-C Analytical Preview")
    lam   = sim_cfg.arrival_rate_per_minute
    svc   = sim_cfg.mean_service_minutes + sim_cfg.acw_mean_minutes
    total = sum(sim_cfg.skill_mix.values()) or 1.0

    print(f"  Arrival rate : {lam:.4f} calls/min  ({ARRIVAL_RATE_PER_HOUR:.0f}/hr)")
    print(f"  Mean svc+ACW : {svc:.2f} min")
    print(f"  SLA target   : {SLA_TARGET:.0%} answered within {SLA_THRESHOLD_MINUTES:.1f} min")
    print(f"  Horizon      : {HORIZON_MINUTES} min  (shifts: M 0–480, D 120–600, E 240–720)")
    print()
    print(f"  {'Skill':<14}  {'λ peak-band':>12}  {'Min agents':>11}  {'SLA @ min':>10}")
    divider()

    from optimization import _arrival_fraction_for_band
    for skill, frac in sim_cfg.skill_mix.items():
        skill_lam = lam * (frac / total)
        peak_lam  = skill_lam * _arrival_fraction_for_band(PEAK_BAND_INDEX)
        min_c     = ErlangC.min_agents_for_sla(peak_lam, svc, SLA_THRESHOLD_MINUTES, SLA_TARGET,
                                                max_c=MAX_AGENTS_PER_SKILL)
        sla_c     = ErlangC.sla_probability(min_c, peak_lam, svc, SLA_THRESHOLD_MINUTES)
        print(f"  {skill:<14}  {peak_lam:>12.4f}  {min_c:>11}  {sla_c:>9.1%}")

    print()
    print("  Note: peak-band λ = full-shift λ × (240/720)")
    print("        Peak band [240–480 min] is where all 3 shifts overlap simultaneously.")

    # Per-band Erlang-C scan
    print()
    print("  Per-band Erlang-C scan (shows SLA floor required at each time band):")
    divider()
    print(f"  {'Band':<22}  {'Shifts':<9}  {'Skill':<12}  {'λ (calls/min)':>14}  {'Min agents':>11}  {'SLA':>7}")
    divider()
    from optimization import _build_band_erlang_mins
    opt_cfg_preview = OptimizationConfig(
        sla_target=SLA_TARGET, sla_threshold_minutes=SLA_THRESHOLD_MINUTES,
        max_agents_per_skill=MAX_AGENTS_PER_SKILL, min_agents_per_skill=MIN_AGENTS_PER_SHIFT,
        arrival_rate_buffer=1.10, max_occupancy=0.85,
    )
    band_mins = _build_band_erlang_mins(sim_cfg, opt_cfg_preview, SLA_TARGET,
                                        lam * 1.10, svc)
    for band_idx, (b_start, b_end, active_shifts) in enumerate(COVERAGE_BANDS):
        sh_str   = "+".join(active_shifts)
        peak_tag = " ← peak" if band_idx == PEAK_BAND_INDEX else ""
        for i, skill in enumerate(sorted(sim_cfg.skill_mix.keys())):
            frac      = sim_cfg.skill_mix[skill] / total
            band_frac = (b_end - b_start) / HORIZON_MINUTES
            skill_lam = lam * 1.10 * band_frac * frac
            min_c     = band_mins[(band_idx, skill)]
            sla_c     = ErlangC.sla_probability(min_c, round(skill_lam, 6), svc, SLA_THRESHOLD_MINUTES)
            band_label = f"[{b_start:>4}–{b_end:>4}m] {sh_str}" if i == 0 else ""
            peak_label = peak_tag if i == 0 else ""
            print(f"  {band_label:<22}  {'':9}  {skill:<12}  {skill_lam:>14.4f}  {min_c:>11}  {sla_c:>6.1%}{peak_label}")


# ============================================================================
# STAGE 1 — Single CP-SAT solve (analytical, no simulation)
# ============================================================================

def stage1_milp_solve() -> dict:
    section("STAGE 1 — Single CP-SAT Solve (Analytical, Multi-Shift)")

    opt_cfg = OptimizationConfig(
        sla_target               = SLA_TARGET,
        sla_threshold_minutes    = SLA_THRESHOLD_MINUTES,
        max_agents_per_skill     = MAX_AGENTS_PER_SKILL,
        min_agents_per_skill     = MIN_AGENTS_PER_SHIFT,   # FIX-E: =1, band constraints are real floor
        analytical_safety_margin = 0.0,
        max_iterations           = 1,
        engine_type              = "cost",
        random_seed              = SEED,
        verbose                  = False,
        arrival_rate_buffer      = 1.10,
        max_occupancy            = 0.85,
        per_shift_optimisation   = True,
    )

    optimizer = StaffingOptimizer(sim_cfg, opt_cfg, cost_cfg)
    result    = optimizer.solve()
    wages     = cost_cfg.hourly_wage_per_skill
    overhead  = cost_cfg.overhead_factor

    print(f"  Solver status  : {result.status}")
    print(f"  Solve time     : {result.solve_time_seconds*1000:.1f} ms")
    print(f"  Total cost     : £{result.total_staffing_cost:,.2f}  (staffing wages only)")
    print_shift_plan(result, wages, overhead)
    print()
    print(f"  Peak-band Erlang-C SLA per skill:")
    divider()
    print(f"  {'Skill':<14}  {'Peak agents':>12}  {'Erlang-C SLA':>13}  {'Target':>8}")
    divider()
    for skill, n in result.agents_per_skill.items():
        sla = result.analytical_sla.get(skill, 0.0)
        tgt = result.analytical_target
        ok  = "✓" if sla >= tgt else "✗"
        print(f"  {skill:<14}  {n:>12}  {sla:>13.1%}  {tgt:>7.1%}  {ok}")
    return result.agents_per_skill


# ============================================================================
# STAGE 2 — Optimize → Simulate loop
# ============================================================================

def stage2_optimize_simulate() -> "tuple[dict, OptimizationResult]":
    section("STAGE 2 — Optimize → Simulate Loop  (Multi-Shift, 720-min horizon)")

    opt_cfg = OptimizationConfig(
        sla_target                     = SLA_TARGET,
        sla_threshold_minutes          = SLA_THRESHOLD_MINUTES,
        max_agents_per_skill           = MAX_AGENTS_PER_SKILL,
        min_agents_per_skill           = MIN_AGENTS_PER_SHIFT,   # FIX-E: =1, band constraints are real floor
        analytical_safety_margin       = None,
        max_iterations                 = MAX_OPT_ITERATIONS,
        convergence_tolerance          = CONVERGENCE_TOLERANCE,
        engine_type                    = "cost",
        random_seed                    = SEED,
        verbose                        = True,
        arrival_rate_buffer            = 1.10,
        max_occupancy                  = 0.85,
        sla_violation_penalty_per_call = 8.0,
        pareto_sweep                   = True,
        sim_feedback_penalty_scale     = 0.5,
        per_shift_optimisation         = True,
    )

    loop    = OptimizeSimulateLoop(sim_cfg, opt_cfg, cost_cfg)
    history = loop.run()

    loop.report.print_report()
    loop.print_pareto_summary()
    loop.explain_plan()

    best_plan  = loop.best_plan
    best_eval  = loop.best_evaluation
    best_entry = loop._best_history_entry()

    section("STAGE 2 — Best Shift Plan Detail")
    if best_entry:
        wages   = cost_cfg.hourly_wage_per_skill
        overhead = cost_cfg.overhead_factor
        print_shift_plan(best_entry.opt_result, wages, overhead)

    section("STAGE 2 — Best Plan KPI Summary")
    print(f"  Peak-band plan     : {best_plan}")
    if best_eval:
        sla_ok = "✓ MEETS TARGET" if best_eval.sla >= SLA_TARGET else "✗ BELOW TARGET"
        print(f"  Simulated SLA      : {best_eval.sla:.1%}  {sla_ok}")
        print(f"  Abandonment rate   : {best_eval.abandonment_rate:.1%}")
        print(f"  Avg CSAT           : {best_eval.avg_csat:.3f}")
        print(f"  Avg speed of answer: {best_eval.asa:.2f} min")
        print(f"  Avg handle time    : {best_eval.aht:.2f} min")
        print(f"  Total calls        : {best_eval.total_calls}")
        print(f"  Total business cost: £{best_eval.total_cost:,.2f}")
        if best_eval.cost_breakdown:
            print()
            print(f"  Cost breakdown:")
            divider()
            for comp, amt in sorted(best_eval.cost_breakdown.items(), key=lambda x: -x[1]):
                pct = amt / best_eval.total_cost * 100 if best_eval.total_cost else 0
                print(f"    {comp:<24}  £{amt:>10,.2f}  ({pct:.1f}%)")

    # Return both peak plan dict and the full OptimizationResult for FIX-9
    opt_result = best_entry.opt_result if best_entry else None
    return best_plan, opt_result


# ============================================================================
# STAGE 3 — ML feedback loop
# ============================================================================

def stage3_ml_feedback(optimised_cfg: SimulationConfig) -> MLModelRegistry:
    section("STAGE 3 — ML Adaptive Feedback Loop")

    registry = MLModelRegistry().warm_start()

    fb_cfg = FeedbackConfig(
        enabled                   = True,
        n_epochs                  = ML_EPOCHS,
        accumulation_strategy     = "cumulative",
        retrain_csat_model        = True,
        retrain_abandonment_model = True,
        retrain_arrival_model     = False,
        convergence_metric        = "sla",
        convergence_delta         = 0.002,
        convergence_patience      = 2,
        verbose                   = True,
    )

    runner  = FeedbackLoopRunner(optimised_cfg, fb_cfg, registry)
    history = runner.run()

    registry.print_model_report()
    FeedbackLoopReport(history, fb_cfg).print_report()

    section("STAGE 3 — Agent Affinity Table (ML-learned)")
    registry.affinity_model.print_affinity_table()

    return registry


# ============================================================================
# STAGE 4 — Full-stack evaluation
# ============================================================================

def stage4_full_stack_eval(
    optimised_cfg: SimulationConfig,
    registry:      MLModelRegistry,
) -> None:
    section("STAGE 4 — Full-Stack Evaluation  (Realism + Cost + ML)")

    print(f"  Peak-band staffing : {optimised_cfg.agents_per_skill}")
    print(f"  Horizon            : {HORIZON_MINUTES} min  (3 overlapping shifts)")
    print(f"  Break schedule     : M@{_SHIFT_BREAK_MIDPOINTS['M']:.0f}m  "
          f"D@{_SHIFT_BREAK_MIDPOINTS['D']:.0f}m  "
          f"E@{_SHIFT_BREAK_MIDPOINTS['E']:.0f}m  ({_BREAK_DURATION_MINUTES:.0f} min each)")
    print(f"  Engine             : CostAwareRealisticsEngine")
    print(f"  Layers active      : fatigue · learning · break variability · cost ledger · burnout")
    print()

    engine = CostAwareRealisticsEngine(
        config   = optimised_cfg,
        cost_cfg = cost_cfg,
        behavior = beh_cfg,
    )
    engine.run()

    section("STAGE 4a — Full KPI Report")
    engine.kpi.report()

    section("STAGE 4a — Business Cost Breakdown")
    engine.cost_function.report("Optimised + Realism + Cost")

    section("STAGE 4a — Agent Performance Tracker")
    engine.router.tracker.print_summary()

    section("STAGE 4a — Human Realism Agent State")
    engine.realism.print_report()

    section("STAGE 4b — ML-Routed Engine (same optimised staffing)")
    print(f"  Routing quality under trained ML models\n")

    ml_engine = MLSimulationEngine(
        config        = optimised_cfg,
        router_factory= registry.make_router,
    )
    ml_engine.run()

    ml_engine.kpi.report()
    registry.affinity_model.print_affinity_table()

    section("STAGE 4 — Side-by-Side: Realism vs ML Engine")
    print(f"  {'Metric':<30}  {'Realism Engine':>16}  {'ML Engine':>16}  {'Delta':>10}")
    divider()
    metrics = [
        ("SLA",                engine.kpi.sla_percentage(),           ml_engine.kpi.sla_percentage(),           "pct"),
        ("Abandonment rate",   engine.kpi.abandonment_rate(),         ml_engine.kpi.abandonment_rate(),         "pct"),
        ("Avg CSAT (1–5)",     engine.kpi.average_csat(),             ml_engine.kpi.average_csat(),             "f2"),
        ("Average Speed of Answer (min)",          engine.kpi.average_speed_of_answer(),  ml_engine.kpi.average_speed_of_answer(),  "f2"),
        ("Average Handle Time (min)",          engine.kpi.average_handle_time(),      ml_engine.kpi.average_handle_time(),      "f2"),
        ("FCR",                engine.kpi.first_call_resolution(),    ml_engine.kpi.first_call_resolution(),    "pct"),
        ("Total calls",        float(engine.kpi.total_calls()),       float(ml_engine.kpi.total_calls()),       "int"),
        ("Total abandonments", float(engine.kpi.total_abandonments()),float(ml_engine.kpi.total_abandonments()),"int"),
    ]
    for label, v_real, v_ml, fmt in metrics:
        delta = v_ml - v_real
        if fmt == "pct":
            print(f"  {label:<30}  {v_real:>16.1%}  {v_ml:>16.1%}  {delta:>+10.1%}")
        elif fmt == "f2":
            print(f"  {label:<30}  {v_real:>16.2f}  {v_ml:>16.2f}  {delta:>+10.2f}")
        else:
            print(f"  {label:<30}  {v_real:>16.0f}  {v_ml:>16.0f}  {delta:>+10.0f}")


# ============================================================================
# STAGE 5 — Scenario comparison: default vs optimised
# ============================================================================

def stage5_scenario_comparison(optimised_cfg: SimulationConfig) -> None:
    section("STAGE 5 — Scenario Comparison: Default vs Optimised Staffing")

    # FIX-10: default config already has the 720-min break schedule from sim_cfg
    default_cfg = copy.copy(sim_cfg)

    print("  Running scenario A — DEFAULT staffing  (11 agents, 720-min horizon) ...")
    eng_a = CostAwareRealisticsEngine(default_cfg, cost_cfg, beh_cfg)
    eng_a.run()
    res_a = ScenarioCostResult.from_engine("Default staffing", eng_a)

    print("  Running scenario B — OPTIMISED staffing (15 agents, 720-min horizon) ...")
    eng_b = CostAwareRealisticsEngine(optimised_cfg, cost_cfg, beh_cfg)
    eng_b.run()
    res_b = ScenarioCostResult.from_engine("Optimised staffing", eng_b)

    CostReport([res_a, res_b]).print_report()

    section("STAGE 5 — KPI Delta (Default → Optimised)")
    sla_delta  = eng_b.kpi.sla_percentage()          - eng_a.kpi.sla_percentage()
    abn_delta  = eng_b.kpi.abandonment_rate()        - eng_a.kpi.abandonment_rate()
    csat_delta = eng_b.kpi.average_csat()            - eng_a.kpi.average_csat()
    asa_delta  = eng_b.kpi.average_speed_of_answer() - eng_a.kpi.average_speed_of_answer()
    cost_delta = res_b.total                         - res_a.total

    rows = [
        ("SLA",                eng_a.kpi.sla_percentage(),          eng_b.kpi.sla_percentage(),          sla_delta,  True,  ".1%"),
        ("Abandonment rate",   eng_a.kpi.abandonment_rate(),        eng_b.kpi.abandonment_rate(),        abn_delta,  False, ".1%"),
        ("Avg CSAT",           eng_a.kpi.average_csat(),            eng_b.kpi.average_csat(),            csat_delta, True,  ".3f"),
        ("ASA (min)",          eng_a.kpi.average_speed_of_answer(), eng_b.kpi.average_speed_of_answer(), asa_delta,  False, ".2f"),
        ("Total business cost",res_a.total,                         res_b.total,                         cost_delta, False, ",.2f"),
    ]
    print(f"  {'Metric':<24}  {'Default':>14}  {'Optimised':>14}  {'Delta':>12}  {'Result':>10}")
    divider()
    for label, v_a, v_b, delta, higher_better, fmt in rows:
        improved = (delta > 0) == higher_better
        tag      = "✓ better" if improved else ("✗ worse" if abs(delta) > 1e-6 else "= same")
        sign     = "+" if delta >= 0 else ""
        if fmt == ",.2f":
            print(f"  {label:<24}  £{v_a:>12,.2f}  £{v_b:>12,.2f}  {sign}£{abs(delta):>9,.2f}  {tag:>10}")
        else:
            print(f"  {label:<24}  {v_a:>{14}{fmt}}  {v_b:>{14}{fmt}}  {sign}{abs(delta):>{12}{fmt}}  {tag:>10}")

    print()
    saving = res_a.total - res_b.total
    if saving > 0:
        print(f"  💰 Net saving with optimised plan: £{saving:,.2f} per {HORIZON_MINUTES}-min horizon")
    else:
        print(f"  ⚠  Optimised plan costs £{abs(saving):,.2f} more (justified by higher SLA & lower SLA-violation cost)")


# ============================================================================
# MASTER RUN
# ============================================================================

def main() -> None:
    t_start = time.perf_counter()

    section("CALL CENTRE STAFFING PIPELINE — FULL RUN", width=72)
    print(f"  SLA target   : {SLA_TARGET:.0%} within {SLA_THRESHOLD_MINUTES:.1f} min")
    print(f"  Arrival rate : {ARRIVAL_RATE_PER_HOUR:.0f} calls/hr")
    print(f"  Horizon      : {HORIZON_MINUTES} min  (3 overlapping 8-hr shifts)")
    print(f"    Shift M    :   0 – 480 min  (Morning)   break @ {_SHIFT_BREAK_MIDPOINTS['M']:.0f} min")
    print(f"    Shift D    : 120 – 600 min  (Mid-Day)   break @ {_SHIFT_BREAK_MIDPOINTS['D']:.0f} min")
    print(f"    Shift E    : 240 – 720 min  (Evening)   break @ {_SHIFT_BREAK_MIDPOINTS['E']:.0f} min")
    print(f"  Random seed  : {SEED}")
    print(f"  Skill mix    : {sim_cfg.skill_mix}")
    print(f"  Default pool : {sim_cfg.agents_per_skill}")
    print(f"  Min agents / shift : {MIN_AGENTS_PER_SHIFT}  (band Erlang-C constraints are the real SLA floor)")
    print()
    print(f"  Stages:")
    print(f"    0 — Erlang-C analytical preview  (per-band scan)")
    print(f"    1 — Single CP-SAT solve  (analytical, full shift plan)")
    print(f"    2 — Optimize → Simulate loop  (max {MAX_OPT_ITERATIONS} iterations)")
    print(f"    3 — ML feedback loop  ({ML_EPOCHS} epochs)")
    print(f"    4 — Full-stack evaluation  (realism + cost + ML)")
    print(f"    5 — Scenario comparison  (default vs optimised)")

    stage0_erlang_preview()
    milp_plan = stage1_milp_solve()

    # FIX-8+9: stage2 now returns (peak_plan, full OptimizationResult)
    optimised_plan, best_opt_result = stage2_optimize_simulate()

    # FIX-9: build Stage 4 config with shift-aware break schedule
    if best_opt_result is not None:
        optimised_cfg = build_optimised_sim_cfg(sim_cfg, best_opt_result)
    else:
        optimised_cfg = copy.copy(sim_cfg)
        optimised_cfg.agents_per_skill     = dict(optimised_plan)
        optimised_cfg.sim_duration_minutes = float(HORIZON_MINUTES)
        optimised_cfg.break_schedule       = [
            (_SHIFT_BREAK_MIDPOINTS["M"], _BREAK_DURATION_MINUTES),
            (_SHIFT_BREAK_MIDPOINTS["D"], _BREAK_DURATION_MINUTES),
            (_SHIFT_BREAK_MIDPOINTS["E"], _BREAK_DURATION_MINUTES),
        ]

    registry = stage3_ml_feedback(optimised_cfg)
    stage4_full_stack_eval(optimised_cfg, registry)
    stage5_scenario_comparison(optimised_cfg)

    elapsed = time.perf_counter() - t_start
    section("PIPELINE COMPLETE", width=72)
    print(f"  Optimised peak-band plan : {optimised_plan}")
    if best_opt_result:
        print(f"  Full shift plan:")
        for sh_name, sh_start, sh_end in SHIFT_WINDOWS:
            plan = best_opt_result.shift_plan.get(sh_name, {})
            print(f"    Shift {sh_name} ({sh_start:>4}–{sh_end:>4}m): {plan}")
    print(f"  Horizon                  : {HORIZON_MINUTES} min")
    print(f"  Total runtime            : {elapsed:.1f}s")
    print()


if __name__ == "__main__":
    main()