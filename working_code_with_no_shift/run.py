"""
main.py  (MULTI-SHIFT FIXED)
=============================
Changes vs original:
  FIX-1  SIM_DURATION_MINUTES → 720 (matches HORIZON_MINUTES in optimization_simpy.py)
  FIX-2  Stage 1 prints shift_plan, band_coverage, and per-shift costs
  FIX-3  Stage 2 explicitly prints shift plan after convergence
  FIX-4  Stage 4 config uses 720-min horizon
  FIX-5  Stage 5 both scenarios run at 720 min
  FIX-6  Pipeline header shows horizon correctly
  FIX-7  New helper print_shift_plan() shared across stages
"""

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
    """
    Pretty-print the full multi-shift plan: per-shift headcount, per-shift
    cost, and per-band coverage summary.  Call this wherever shift data
    should be visible.
    """
    SHIFT_HOURS = 8.0

    print(f"\n  ┌─ Shift Plan  ({HORIZON_MINUTES}-min horizon: 3 overlapping 8-hr shifts) ─────┐")
    for sh_name, sh_start, sh_end in SHIFT_WINDOWS:
        plan = opt_result.shift_plan.get(sh_name, {})
        shift_cost = sum(
            plan.get(sk, 0) * wages.get(sk, 18.0) * overhead * SHIFT_HOURS
            for sk in plan
        )
        agents_str = "  ".join(f"{sk}: {n}" for sk, n in sorted(plan.items()))
        print(f"  │  Shift {sh_name}  ({sh_start:>4}–{sh_end:>4} min)  [{agents_str}]"
              f"  cost=£{shift_cost:,.2f}")
    print(f"  └" + "─" * 68)

    # Band coverage table
    skills = sorted(opt_result.agents_per_skill.keys())
    hdr_skills = "".join(f"  {s[:8]:>9}" for s in skills)
    print(f"\n  {'Band interval':<22}  {'Shifts':>9}{hdr_skills}")
    divider(22 + 12 + 10 * len(skills))
    for band_idx, (b_start, b_end, active_shifts) in enumerate(COVERAGE_BANDS):
        cov       = opt_result.band_coverage.get(band_idx, {})
        sh_str    = "+".join(active_shifts)
        agents_s  = "".join(f"  {cov.get(sk, 0):>9}" for sk in skills)
        peak_tag  = "  ← PEAK" if band_idx == PEAK_BAND_INDEX else ""
        print(f"  [{b_start:>4}–{b_end:>4} min]        {sh_str:>9}{agents_s}{peak_tag}")


# ============================================================================
# CONFIGURATION
# ============================================================================

SEED                   = 42
# FIX-1: horizon must match HORIZON_MINUTES (720) so SimulationEvaluator
#         runs the full 3-shift window, not just the morning shift.
SIM_DURATION_MINUTES   = float(HORIZON_MINUTES)   # 720 min = 12 hr planning window
ARRIVAL_RATE_PER_HOUR  = 60.0
SLA_TARGET             = 0.90
SLA_THRESHOLD_MINUTES  = 1.0
MAX_AGENTS_PER_SKILL   = 15
MAX_OPT_ITERATIONS     = 8
CONVERGENCE_TOLERANCE  = 0.015
ML_EPOCHS              = 5

sim_cfg  = SimulationConfig(
    sim_duration_minutes  = SIM_DURATION_MINUTES,   # FIX-1
    arrival_rate_per_hour = ARRIVAL_RATE_PER_HOUR,
    random_seed           = SEED,
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
    print(f"  {'Skill':<14}  {'λ (calls/min)':>14}  {'Min agents (peak)':>18}  {'SLA @ min':>10}")
    divider()

    for skill, frac in sim_cfg.skill_mix.items():
        skill_lam = lam * (frac / total)
        # Use peak-band fraction for the analytical preview
        from optimization import _arrival_fraction_for_band
        peak_lam = skill_lam * _arrival_fraction_for_band(PEAK_BAND_INDEX)
        min_c = ErlangC.min_agents_for_sla(
            peak_lam, svc, SLA_THRESHOLD_MINUTES, SLA_TARGET,
            max_c=MAX_AGENTS_PER_SKILL,
        )
        sla_c = ErlangC.sla_probability(min_c, peak_lam, svc, SLA_THRESHOLD_MINUTES)
        print(f"  {skill:<14}  {peak_lam:>14.4f}  {min_c:>18}  {sla_c:>9.1%}")

    print()
    print("  Note: peak-band λ = full-shift λ × (240/720) — fraction of daily load")
    print("        in the 240–480 min band where all three shifts overlap.")


# ============================================================================
# STAGE 1 — Single CP-SAT solve (analytical, no simulation)
# ============================================================================

def stage1_milp_solve() -> dict:
    section("STAGE 1 — Single CP-SAT Solve (Analytical, Multi-Shift)")

    opt_cfg = OptimizationConfig(
        sla_target               = SLA_TARGET,
        sla_threshold_minutes    = SLA_THRESHOLD_MINUTES,
        max_agents_per_skill     = MAX_AGENTS_PER_SKILL,
        min_agents_per_skill     = 1,
        analytical_safety_margin = 0.0,
        max_iterations           = 1,
        engine_type              = "cost",
        random_seed              = SEED,
        verbose                  = False,
        arrival_rate_buffer      = 1.10,
        max_occupancy            = 0.85,
        per_shift_optimisation   = True,   # FIX-2: enable multi-shift mode
    )

    optimizer = StaffingOptimizer(sim_cfg, opt_cfg, cost_cfg)
    result    = optimizer.solve()

    wages    = cost_cfg.hourly_wage_per_skill
    overhead = cost_cfg.overhead_factor

    print(f"  Solver status  : {result.status}")
    print(f"  Solve time     : {result.solve_time_seconds*1000:.1f} ms")
    print(f"  Total cost     : £{result.total_staffing_cost:,.2f}  (staffing wages only)")
    print()

    # FIX-2: show full shift plan
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

def stage2_optimize_simulate() -> dict:
    section("STAGE 2 — Optimize → Simulate Loop  (Multi-Shift, 720-min horizon)")

    opt_cfg = OptimizationConfig(
        sla_target                     = SLA_TARGET,
        sla_threshold_minutes          = SLA_THRESHOLD_MINUTES,
        max_agents_per_skill           = MAX_AGENTS_PER_SKILL,
        min_agents_per_skill           = 1,
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
        per_shift_optimisation         = True,   # FIX-3
    )

    loop    = OptimizeSimulateLoop(sim_cfg, opt_cfg, cost_cfg)
    history = loop.run()

    loop.report.print_report()
    loop.print_pareto_summary()
    loop.explain_plan()          # already prints shift plan via explain_plan()

    best_plan = loop.best_plan
    best_eval = loop.best_evaluation

    # FIX-3: explicitly print the winning shift plan
    section("STAGE 2 — Best Shift Plan Detail")
    entry = loop._best_history_entry()
    if entry:
        wages    = cost_cfg.hourly_wage_per_skill
        overhead = cost_cfg.overhead_factor
        print_shift_plan(entry.opt_result, wages, overhead)

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

    return best_plan


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
        ("SLA",                engine.kpi.sla_percentage(),           ml_engine.kpi.sla_percentage(),          "pct"),
        ("Abandonment rate",   engine.kpi.abandonment_rate(),         ml_engine.kpi.abandonment_rate(),        "pct"),
        ("Avg CSAT (1–5)",     engine.kpi.average_csat(),             ml_engine.kpi.average_csat(),            "f2"),
        ("ASA (min)",          engine.kpi.average_speed_of_answer(),  ml_engine.kpi.average_speed_of_answer(), "f2"),
        ("AHT (min)",          engine.kpi.average_handle_time(),      ml_engine.kpi.average_handle_time(),     "f2"),
        ("FCR",                engine.kpi.first_call_resolution(),    ml_engine.kpi.first_call_resolution(),   "pct"),
        ("Total calls",        float(engine.kpi.total_calls()),       float(ml_engine.kpi.total_calls()),      "int"),
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

    # FIX-5: run both scenarios at 720-min horizon
    default_cfg = copy.copy(sim_cfg)   # already 720 min from SIM_DURATION_MINUTES fix

    print("  Running scenario A — DEFAULT staffing ...")
    eng_a = CostAwareRealisticsEngine(default_cfg, cost_cfg, beh_cfg)
    eng_a.run()
    res_a = ScenarioCostResult.from_engine("Default staffing", eng_a)

    print("  Running scenario B — OPTIMISED staffing ...")
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
        print(f"  ⚠  Optimised plan costs £{abs(saving):,.2f} more (justified by higher SLA)")


# ============================================================================
# MASTER RUN
# ============================================================================

def main() -> None:
    t_start = time.perf_counter()

    section("CALL CENTRE STAFFING PIPELINE — FULL RUN", width=72)
    print(f"  SLA target   : {SLA_TARGET:.0%} within {SLA_THRESHOLD_MINUTES:.1f} min")
    print(f"  Arrival rate : {ARRIVAL_RATE_PER_HOUR:.0f} calls/hr")
    print(f"  Horizon      : {HORIZON_MINUTES} min  (3 overlapping 8-hr shifts)")
    print(f"    Shift M    :   0 – 480 min  (Morning)")
    print(f"    Shift D    : 120 – 600 min  (Mid-Day)")
    print(f"    Shift E    : 240 – 720 min  (Evening)")
    print(f"  Random seed  : {SEED}")
    print(f"  Skill mix    : {sim_cfg.skill_mix}")
    print(f"  Default pool : {sim_cfg.agents_per_skill}")
    print()
    print(f"  Stages:")
    print(f"    0 — Erlang-C analytical preview  (peak-band λ)")
    print(f"    1 — Single CP-SAT solve  (analytical, full shift plan)")
    print(f"    2 — Optimize → Simulate loop  (max {MAX_OPT_ITERATIONS} iterations)")
    print(f"    3 — ML feedback loop  ({ML_EPOCHS} epochs)")
    print(f"    4 — Full-stack evaluation  (realism + cost + ML)")
    print(f"    5 — Scenario comparison  (default vs optimised)")

    stage0_erlang_preview()

    milp_plan      = stage1_milp_solve()
    optimised_plan = stage2_optimize_simulate()

    # FIX-4: build optimised config at 720-min horizon
    optimised_cfg = copy.copy(sim_cfg)
    optimised_cfg.agents_per_skill     = dict(optimised_plan)
    optimised_cfg.sim_duration_minutes = float(HORIZON_MINUTES)

    registry = stage3_ml_feedback(optimised_cfg)
    stage4_full_stack_eval(optimised_cfg, registry)
    stage5_scenario_comparison(optimised_cfg)

    elapsed = time.perf_counter() - t_start
    section("PIPELINE COMPLETE", width=72)
    print(f"  Optimised peak-band plan : {optimised_plan}")
    print(f"  Horizon                  : {HORIZON_MINUTES} min")
    print(f"  Total runtime            : {elapsed:.1f}s")
    print()


if __name__ == "__main__":
    main()