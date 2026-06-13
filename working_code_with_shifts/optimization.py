from __future__ import annotations

import copy
import functools
import logging
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ortools.sat.python import cp_model

try:
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

from core_simulation import RouterScoreWeights, SimulationConfig, SimulationEngine

logger = logging.getLogger(__name__)

_COST_SCALE: int = 100

HORIZON_MINUTES: int = 720

SHIFT_WINDOWS: Tuple[Tuple[str, int, int], ...] = (
    ("M", 0,   480),
    ("D", 120, 600),
    ("E", 240, 720),
)

COVERAGE_BANDS: Tuple[Tuple[int, int, Tuple[str, ...]], ...] = (
    (  0, 120, ("M",)),
    (120, 240, ("M", "D")),
    (240, 480, ("M", "D", "E")),
    (480, 600, ("D", "E")),
    (600, 720, ("E",)),
)

PEAK_BAND_INDEX: int = 2

# FIX-A: one break per shift, anchored to that shift's midpoint.
# Shift M midpoint = (0+480)/2 = 240
# Shift D midpoint = (120+600)/2 = 360
# Shift E midpoint = (240+720)/2 = 480
_SHIFT_BREAK_MIDPOINTS: Dict[str, float] = {
    "M": 240.0,
    "D": 360.0,
    "E": 480.0,
}
_BREAK_DURATION_MINUTES: float = 30.0


def _shifts_for_band(band_idx: int) -> Tuple[str, ...]:
    return COVERAGE_BANDS[band_idx][2]


def _band_duration(band_idx: int) -> int:
    start, end, _ = COVERAGE_BANDS[band_idx]
    return end - start


def _arrival_fraction_for_band(band_idx: int) -> float:
    return _band_duration(band_idx) / HORIZON_MINUTES


# ---------------------------------------------------------------------------
# OptimizationConfig
# ---------------------------------------------------------------------------

@dataclass
class OptimizationConfig:
    sla_target:                     float           = 0.97
    sla_threshold_minutes:          float           = 1.0
    max_agents_per_skill:           int             = 20
    max_agents_per_skill:           int             = 20
    # FIX-E: reset to 1 — band Erlang-C constraints are the real floor.
    # min=2 forced symmetric 2-2-2 plans, over-staffing peak from 3→6.
    min_agents_per_skill:           int             = 1
    analytical_safety_margin:       Optional[float] = 0.04
    max_iterations:                 int             = 8
    convergence_tolerance:          float           = 0.015
    engine_type:                    str             = "cost"
    random_seed:                    Optional[int]   = 42
    verbose:                        bool            = True
    sla_violation_penalty_per_call: float           = 0.0
    cost_weight:                    float           = 1.0
    sla_weight:                     float           = 0.0
    pareto_sweep:                   bool            = False
    max_total_agents:               Optional[int]   = None
    max_occupancy:                  float           = 0.85
    arrival_rate_buffer:            float           = 1.10
    debug_solver:                   bool            = False
    sla_predictor:                  Optional[Callable] = None
    sim_feedback_penalty_scale:     float           = 0.0
    skill_realism_derating:         Dict[str, float] = field(default_factory=dict)
    realism_floor_agents:           Dict[str, int]   = field(default_factory=dict)
    cpsat_time_limit_seconds:       float           = 30.0
    per_shift_optimisation:         bool            = True


# ---------------------------------------------------------------------------
# ErlangC
# ---------------------------------------------------------------------------

class ErlangC:
    @staticmethod
    @functools.lru_cache(maxsize=4096)
    def erlang_c_probability(c: int, a: float) -> float:
        if c <= 0:
            return 1.0
        rho = a / c
        if rho >= 1.0:
            return 1.0
        log_a        = math.log(a) if a > 0 else float("-inf")
        log_num_term = c * log_a - math.lgamma(c + 1)
        num_term     = math.exp(log_num_term) / (1.0 - rho)
        poisson_sum  = 0.0
        log_ak_kfact = 0.0
        for k in range(c):
            if k > 0:
                log_ak_kfact += log_a - math.log(k)
            poisson_sum += math.exp(log_ak_kfact)
        denominator = poisson_sum + num_term
        return num_term / denominator if denominator > 0 else 1.0

    @staticmethod
    @functools.lru_cache(maxsize=4096)
    def sla_probability(
        c:                    int,
        arrival_rate_per_min: float,
        mean_service_min:     float,
        sla_threshold_min:    float,
    ) -> float:
        if c <= 0 or mean_service_min <= 0:
            return 0.0
        mu  = 1.0 / mean_service_min
        a   = arrival_rate_per_min / mu
        if a <= 0:
            return 1.0
        C   = ErlangC.erlang_c_probability(c, round(a, 6))
        rho = a / c
        if rho >= 1.0:
            return 0.0
        exponent = -(c - a) * mu * sla_threshold_min
        sla      = 1.0 - C * math.exp(exponent)
        return float(max(0.0, min(1.0, sla)))

    @staticmethod
    def min_agents_for_sla(
        arrival_rate_per_min: float,
        mean_service_min:     float,
        sla_threshold_min:    float,
        sla_target:           float,
        max_c:                int = 50,
        sla_predictor:        Optional[Callable] = None,
    ) -> int:
        mu    = 1.0 / mean_service_min if mean_service_min > 0 else 1.0
        a     = arrival_rate_per_min / mu
        c_min = max(1, math.ceil(a) + 1)
        lam_r = round(arrival_rate_per_min, 6)
        svc_r = round(mean_service_min, 6)
        thr_r = round(sla_threshold_min, 6)
        for c in range(c_min, max_c + 1):
            try:
                sla = (
                    float(sla_predictor(c, lam_r, svc_r, thr_r))
                    if sla_predictor is not None
                    else ErlangC.sla_probability(c, lam_r, svc_r, thr_r)
                )
            except Exception:
                sla = ErlangC.sla_probability(c, lam_r, svc_r, thr_r)
            if sla >= sla_target:
                return c
        return max_c


# ---------------------------------------------------------------------------
# OptimizationResult
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    agents_per_skill:    Dict[str, int]
    shift_plan:          Dict[str, Dict[str, int]]
    band_coverage:       Dict[int, Dict[str, int]]
    analytical_sla:      Dict[str, float]
    analytical_target:   float
    total_staffing_cost: float
    status:              str
    solve_time_seconds:  float = 0.0

    def to_sim_config(self, base: SimulationConfig) -> SimulationConfig:
        cfg = copy.copy(base)
        cfg.agents_per_skill     = dict(self.agents_per_skill)
        cfg.sim_duration_minutes = float(HORIZON_MINUTES)
        return cfg

    def __repr__(self) -> str:
        lines = [
            f"OptimizationResult(status={self.status}, "
            f"cost=£{self.total_staffing_cost:,.2f}, "
            f"horizon={HORIZON_MINUTES}min)"
        ]
        for skill, n in self.agents_per_skill.items():
            sla = self.analytical_sla.get(skill, 0.0)
            lines.append(f"  {skill}: {n} agents (peak)  Erlang-C SLA={sla:.1%}")
        lines.append("  Shift plan:")
        for sh, plan in self.shift_plan.items():
            lines.append(f"    Shift {sh}: {plan}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    agents_per_skill:  Dict[str, int]
    shift_plan:        Dict[str, Dict[str, int]]
    sla:               float
    abandonment_rate:  float
    avg_csat:          float
    asa:               float
    aht:               float
    total_calls:       int
    total_cost:        float = 0.0
    cost_breakdown:    Dict[str, float] = field(default_factory=dict)
    sim_time_seconds:  float = 0.0

    def meets_sla(self, target: float, tolerance: float = 0.0) -> bool:
        return self.sla >= target - tolerance

    def __repr__(self) -> str:
        return (
            f"EvaluationResult(SLA={self.sla:.1%}, "
            f"abandon={self.abandonment_rate:.1%}, "
            f"CSAT={self.avg_csat:.2f}, "
            f"cost=£{self.total_cost:,.2f}, "
            f"horizon={HORIZON_MINUTES}min)"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_derated_predictor(skill: str, derating: Dict[str, float]) -> Callable:
    factor = float(derating.get(skill, 1.0))
    def predictor(c: int, lam: float, svc: float, thr: float) -> float:
        return ErlangC.sla_probability(c, lam, svc, thr) * factor
    return predictor


def _build_band_erlang_mins(
    sim_cfg: SimulationConfig,
    opt_cfg: OptimizationConfig,
    analytical_target: float,
    lam_buffered: float,
    mean_svc: float,
) -> Dict[Tuple[int, str], int]:
    skills    = list(sim_cfg.agents_per_skill.keys())
    skill_mix = sim_cfg.skill_mix
    total_mix = sum(skill_mix.values()) or 1.0
    n         = len(skills)
    result: Dict[Tuple[int, str], int] = {}

    for band_idx, (b_start, b_end, _) in enumerate(COVERAGE_BANDS):
        band_frac = (b_end - b_start) / HORIZON_MINUTES
        band_lam  = lam_buffered * band_frac

        for skill in skills:
            frac      = skill_mix.get(skill, 1.0 / n) / total_mix
            skill_lam = band_lam * frac

            derating = opt_cfg.skill_realism_derating or {}
            if skill in derating and derating[skill] != 1.0:
                skill_pred: Optional[Callable] = _make_derated_predictor(skill, derating)
            else:
                skill_pred = opt_cfg.sla_predictor

            min_c = ErlangC.min_agents_for_sla(
                arrival_rate_per_min=round(skill_lam, 6),
                mean_service_min    =round(mean_svc, 6),
                sla_threshold_min   =opt_cfg.sla_threshold_minutes,
                sla_target          =analytical_target,
                max_c               =opt_cfg.max_agents_per_skill,
                sla_predictor       =skill_pred,
            )

            mu       = 1.0 / mean_svc if mean_svc > 0 else 1.0
            offered  = skill_lam / mu
            max_occ  = min(0.99, float(opt_cfg.max_occupancy))
            if max_occ > 0 and offered > 0:
                min_c = max(min_c, math.ceil(offered / max_occ))

            floor = int((opt_cfg.realism_floor_agents or {}).get(skill, 0))
            min_c = min(min_c + floor, opt_cfg.max_agents_per_skill)
            min_c = max(min_c, opt_cfg.min_agents_per_skill)

            result[(band_idx, skill)] = min_c

    return result


def _band_coverage_from_shift_plan(
    shift_plan: Dict[str, Dict[str, int]],
    skills: List[str],
) -> Dict[int, Dict[str, int]]:
    band_coverage: Dict[int, Dict[str, int]] = {}
    for band_idx, (_, _, active_shifts) in enumerate(COVERAGE_BANDS):
        band_coverage[band_idx] = {
            skill: sum(
                shift_plan.get(sh, {}).get(skill, 0)
                for sh in active_shifts
            )
            for skill in skills
        }
    return band_coverage


# ---------------------------------------------------------------------------
# StaffingOptimizer
# ---------------------------------------------------------------------------

class StaffingOptimizer:
    _DEFAULT_WAGES    = {"billing": 18.0, "technical": 22.0, "general": 16.0}
    _DEFAULT_OVERHEAD = 1.30
    _SHIFT_HOURS      = 8.0

    _CPSAT_STATUS: Dict[int, str] = {
        cp_model.OPTIMAL:       "optimal",
        cp_model.FEASIBLE:      "feasible",
        cp_model.INFEASIBLE:    "infeasible",
        cp_model.UNKNOWN:       "unknown",
        cp_model.MODEL_INVALID: "model_invalid",
    }

    def __init__(
        self,
        sim_cfg:               SimulationConfig,
        opt_cfg:               OptimizationConfig,
        cost_cfg=None,
        analytical_sla_target: Optional[float] = None,
    ) -> None:
        self._sim    = sim_cfg
        self._opt    = opt_cfg
        self._cost   = cost_cfg
        self._target = (
            analytical_sla_target
            if analytical_sla_target is not None
            else opt_cfg.sla_target
        )
        self.status:  str                          = "not_solved"
        self.value:   Optional[Dict[str, int]]     = None
        self._result: Optional[OptimizationResult] = None
        self._last_sim_feedback: Optional[Dict[str, float]] = None

    def _cost_params(self) -> Tuple[Dict[str, float], float]:
        try:
            return self._cost.hourly_wage_per_skill, self._cost.overhead_factor
        except AttributeError:
            return self._DEFAULT_WAGES, self._DEFAULT_OVERHEAD

    def _agent_cost(self, skill: str, wages: Dict[str, float], overhead: float) -> float:
        return wages.get(skill, self._DEFAULT_WAGES.get(skill, 18.0)) * overhead * self._SHIFT_HOURS

    def solve(self) -> OptimizationResult:
        t0     = time.perf_counter()
        skills = list(self._sim.agents_per_skill.keys())
        n      = len(skills)
        shifts = [name for name, _, _ in SHIFT_WINDOWS]

        wages, overhead  = self._cost_params()
        buffer           = max(1.0, float(self._opt.arrival_rate_buffer))
        lam_per_min_base = self._sim.arrival_rate_per_minute
        lam_buffered     = lam_per_min_base * buffer
        mean_svc         = self._sim.mean_service_minutes + self._sim.acw_mean_minutes
        skill_mix        = self._sim.skill_mix
        total_mix        = sum(skill_mix.values()) or 1.0

        fb_scale = 1.0
        if self._opt.sim_feedback_penalty_scale > 0.0 and self._last_sim_feedback:
            last_gap = self._last_sim_feedback.get("sla_gap", 0.0)
            abn      = self._last_sim_feedback.get("abandonment_rate", 0.0)
            if last_gap < 0:
                fb_scale += abs(last_gap) * float(self._opt.sim_feedback_penalty_scale)
            fb_scale *= 1.0 + abn * float(self._opt.sim_feedback_penalty_scale)

        band_erlang_min = _build_band_erlang_mins(
            self._sim, self._opt, self._target, lam_buffered, mean_svc,
        )

        cost_per_agent: Dict[str, float] = {
            skill: self._agent_cost(skill, wages, overhead) for skill in skills
        }
        penalty  = float(self._opt.sla_violation_penalty_per_call)
        peak_min = {skill: band_erlang_min[(PEAK_BAND_INDEX, skill)] for skill in skills}
        adjusted_cost: Dict[str, float] = dict(cost_per_agent)
        if penalty > 0.0:
            calls_per_shift = lam_per_min_base * self._SHIFT_HOURS * 60
            for skill in skills:
                frac           = skill_mix.get(skill, 1.0 / n) / total_mix
                skill_lam_peak = lam_buffered * _arrival_fraction_for_band(PEAK_BAND_INDEX) * frac
                sla_at_peak    = ErlangC.sla_probability(
                    peak_min[skill], round(skill_lam_peak, 6),
                    round(mean_svc, 6), self._opt.sla_threshold_minutes,
                )
                sla_gap_proxy   = max(0.0, self._target - sla_at_peak)
                skill_calls     = calls_per_shift * frac
                adjusted_cost[skill] += (
                    penalty * fb_scale * sla_gap_proxy * skill_calls
                )

        obj_coeff: Dict[str, int] = {
            skill: int(round(adjusted_cost[skill] * _COST_SCALE))
            for skill in skills
        }

        model = cp_model.CpModel()

        agents: Dict[str, Dict[str, cp_model.IntVar]] = {
            sh: {
                skill: model.NewIntVar(
                    0, self._opt.max_agents_per_skill, f"n_{sh}_{skill}"
                )
                for skill in skills
            }
            for sh in shifts
        }

        # FIX-E: per-shift minimum = 1 (prevents zero-staffed shifts).
        # The band Erlang-C constraints below are the real SLA floor.
        # min=1 allows the solver to produce asymmetric shift plans
        # (e.g. Shift D with fewer agents than M/E) rather than being
        # forced into a symmetric 2-2-2 plan that over-staffs the peak.
        for sh in shifts:
            for skill in skills:
                model.Add(agents[sh][skill] >= self._opt.min_agents_per_skill)

        cover: Dict[int, Dict[str, cp_model.IntVar]] = {}
        for band_idx, (_, _, active_shifts) in enumerate(COVERAGE_BANDS):
            cover[band_idx] = {}
            for skill in skills:
                cv = model.NewIntVar(0, self._opt.max_agents_per_skill * len(shifts),
                                     f"cover_b{band_idx}_{skill}")
                model.Add(cv == sum(agents[sh][skill] for sh in active_shifts))
                cover[band_idx][skill] = cv

        # Band-level SLA floor: each band must meet its Erlang-C minimum.
        # This is the primary staffing constraint — it drives all headcount
        # decisions by ensuring coverage >= Erlang-C minimum in every band.
        for band_idx in range(len(COVERAGE_BANDS)):
            for skill in skills:
                floor = band_erlang_min[(band_idx, skill)]
                model.Add(cover[band_idx][skill] >= floor)

        # FIX-E: Explicit peak-band minimum as a redundant but clear constraint.
        # Ensures the peak band (all 3 shifts active) always meets or exceeds
        # the Erlang-C floor, even if band_erlang_min is later refactored.
        for skill in skills:
            peak_floor = band_erlang_min[(PEAK_BAND_INDEX, skill)]
            model.Add(cover[PEAK_BAND_INDEX][skill] >= peak_floor)

        if self._opt.max_total_agents is not None:
            model.Add(
                sum(agents[sh][skill] for sh in shifts for skill in skills)
                <= int(self._opt.max_total_agents)
            )

        model.Minimize(
            sum(obj_coeff[skill] * agents[sh][skill] for sh in shifts for skill in skills)
        )

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self._opt.cpsat_time_limit_seconds
        solver.parameters.num_search_workers  = 8
        if self._opt.random_seed is not None:
            solver.parameters.random_seed = int(self._opt.random_seed)
        if self._opt.debug_solver:
            solver.parameters.log_search_progress = True

        cpsat_status = solver.Solve(model)
        status_str   = self._CPSAT_STATUS.get(cpsat_status, f"cpsat_{cpsat_status}")

        if cpsat_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            shift_plan: Dict[str, Dict[str, int]] = {
                sh: {skill: solver.Value(agents[sh][skill]) for skill in skills}
                for sh in shifts
            }
            status_out = status_str
            logger.info(
                "CP-SAT %s: shift_plan=%s  obj=£%.2f  wall=%.1fms",
                status_str, shift_plan,
                solver.ObjectiveValue() / _COST_SCALE,
                solver.WallTime() * 1000,
            )
        else:
            logger.warning("CP-SAT %s — using Erlang-C peak-band fallback.", status_str)
            shift_plan = {
                sh: {skill: band_erlang_min[(PEAK_BAND_INDEX, skill)] for skill in skills}
                for sh in shifts
            }
            status_out = f"{status_str}_fallback"

        band_coverage    = _band_coverage_from_shift_plan(shift_plan, skills)
        agents_per_skill = dict(band_coverage[PEAK_BAND_INDEX])

        peak_frac = _arrival_fraction_for_band(PEAK_BAND_INDEX)
        analytical_sla: Dict[str, float] = {}
        for skill in skills:
            frac      = skill_mix.get(skill, 1.0 / n) / total_mix
            skill_lam = lam_buffered * peak_frac * frac
            analytical_sla[skill] = ErlangC.sla_probability(
                c                   =agents_per_skill[skill],
                arrival_rate_per_min=round(skill_lam, 6),
                mean_service_min    =round(mean_svc, 6),
                sla_threshold_min   =self._opt.sla_threshold_minutes,
            )

        total_staffing_cost = sum(
            cost_per_agent[skill] * shift_plan[sh][skill]
            for sh in shifts for skill in skills
        )

        elapsed     = time.perf_counter() - t0
        self.status = status_out
        self.value  = agents_per_skill
        self._result = OptimizationResult(
            agents_per_skill   =agents_per_skill,
            shift_plan         =shift_plan,
            band_coverage      =band_coverage,
            analytical_sla     =analytical_sla,
            analytical_target  =self._target,
            total_staffing_cost=total_staffing_cost,
            status             =status_out,
            solve_time_seconds =elapsed,
        )
        return self._result

    @property
    def result(self) -> Optional[OptimizationResult]:
        return self._result


# ---------------------------------------------------------------------------
# SimulationEvaluator
# ---------------------------------------------------------------------------

class SimulationEvaluator:
    """
    FIX-A applied here: _build_config() now writes exactly three break entries,
    one per shift, each at that shift's midpoint.  The core SimulationEngine
    schedules ALL entries for EVERY agent via _schedule_breaks(), which is
    correct: agents whose shift hasn't started yet will simply find the
    first entry's start time has passed and take a ~0-delay no-op, then
    rest properly at the entry matching their own shift window.

    Shift midpoints (all 30-min breaks):
      Shift M (0–480)   → minute 240
      Shift D (120–600) → minute 360
      Shift E (240–720) → minute 480
    """

    def __init__(
        self,
        base_sim_cfg: SimulationConfig,
        opt_cfg:      OptimizationConfig,
        cost_cfg=None,
        beh_cfg=None,
        weights:      Optional[RouterScoreWeights] = None,
    ) -> None:
        self._base    = base_sim_cfg
        self._opt     = opt_cfg
        self._cost    = cost_cfg
        self._beh     = beh_cfg
        self._weights = weights

    def evaluate(self, opt_result: OptimizationResult) -> EvaluationResult:
        t0     = time.perf_counter()
        cfg    = self._build_config(opt_result)
        engine = self._build_engine(cfg)
        engine.run()
        elapsed = time.perf_counter() - t0

        kpi            = engine.kpi
        total_cost     = 0.0
        cost_breakdown = {}

        if hasattr(engine, "cost_function"):
            bk             = engine.cost_function.breakdown()
            total_cost     = bk.get("total", 0.0)
            cost_breakdown = {k: v for k, v in bk.items() if k != "total"}

        return EvaluationResult(
            agents_per_skill=opt_result.agents_per_skill,
            shift_plan      =opt_result.shift_plan,
            sla             =kpi.sla_percentage(),
            abandonment_rate=kpi.abandonment_rate(),
            avg_csat        =kpi.average_csat(),
            asa             =kpi.average_speed_of_answer(),
            aht             =kpi.average_handle_time(),
            total_calls     =kpi.total_calls(),
            total_cost      =total_cost,
            cost_breakdown  =cost_breakdown,
            sim_time_seconds=elapsed,
        )

    def _build_config(self, opt_result: OptimizationResult) -> SimulationConfig:
        """
        FIX-A: One break per shift at that shift's midpoint.
        All agents see all three entries; the engine's _break_process uses
        max(0, start - now) so entries whose time has already passed resolve
        instantly and have no effect.
        """
        cfg = copy.copy(self._base)
        cfg.agents_per_skill     = dict(opt_result.agents_per_skill)
        cfg.sim_duration_minutes = float(HORIZON_MINUTES)

        cfg.break_schedule = [
            (_SHIFT_BREAK_MIDPOINTS["M"], _BREAK_DURATION_MINUTES),  # minute 240
            (_SHIFT_BREAK_MIDPOINTS["D"], _BREAK_DURATION_MINUTES),  # minute 360
            (_SHIFT_BREAK_MIDPOINTS["E"], _BREAK_DURATION_MINUTES),  # minute 480
        ]
        return cfg

    def _build_engine(self, cfg: SimulationConfig):
        engine_type = self._opt.engine_type
        try:
            from cost_system import CostAwareEngine, CostAwareRealisticsEngine
            from core_simulation import BehaviorConfig
            if engine_type == "realism":
                return CostAwareRealisticsEngine(
                    cfg, self._cost, self._beh or BehaviorConfig(), self._weights
                )
            if engine_type == "cost":
                return CostAwareEngine(cfg, self._cost, self._weights)
        except ImportError:
            pass
        return SimulationEngine(cfg, self._weights)


# ---------------------------------------------------------------------------
# _LoopIteration
# ---------------------------------------------------------------------------

@dataclass
class _LoopIteration:
    iteration:         int
    analytical_target: float
    opt_result:        OptimizationResult
    eval_result:       EvaluationResult
    sla_gap:           float
    converged:         bool


# ---------------------------------------------------------------------------
# OptimizeSimulateLoop
# ---------------------------------------------------------------------------

class OptimizeSimulateLoop:
    def __init__(
        self,
        base_sim_cfg: SimulationConfig,
        opt_cfg:      OptimizationConfig,
        cost_cfg=None,
        beh_cfg=None,
        weights:      Optional[RouterScoreWeights] = None,
    ) -> None:
        self._base      = base_sim_cfg
        self._opt       = opt_cfg
        self._cost      = cost_cfg
        self._beh       = beh_cfg
        self._weights   = weights
        self._evaluator = SimulationEvaluator(base_sim_cfg, opt_cfg, cost_cfg, beh_cfg, weights)
        self._history:  List[_LoopIteration] = []
        self.report     = LoopReport(self._history, opt_cfg)
        self.pareto_points: List[Dict] = []

    def run(self) -> List[_LoopIteration]:
        opt_cfg = self._opt
        target  = opt_cfg.sla_target

        if opt_cfg.verbose:
            adaptive_tag = (
                "adaptive"
                if opt_cfg.analytical_safety_margin is None
                else f"fixed={opt_cfg.analytical_safety_margin:.2f}"
            )
            print("\n" + "=" * 72)
            print("  STAFFING OPTIMISATION + SIMULATION LOOP  [OR-Tools CP-SAT]")
            print(f"  Horizon  : {HORIZON_MINUTES} min  |  Shifts: M(0-480) D(120-600) E(240-720)")
            print(f"  SLA tgt  : {opt_cfg.sla_target:.1%}  |  Engine: {opt_cfg.engine_type}"
                  f"  |  max_iter: {opt_cfg.max_iterations}  |  correction: {adaptive_tag}")
            if opt_cfg.skill_realism_derating:
                print(f"  Realism de-rating : {opt_cfg.skill_realism_derating}")
            if opt_cfg.realism_floor_agents:
                print(f"  Realism floor     : {opt_cfg.realism_floor_agents}")
            print("=" * 72)

        _sim_feedback: Optional[Dict[str, float]] = None

        for iteration in range(1, opt_cfg.max_iterations + 1):
            if opt_cfg.verbose:
                print(f"\n  [Iter {iteration}] Analytical SLA target = {target:.3%}")

            optimizer = StaffingOptimizer(
                sim_cfg              =self._base,
                opt_cfg              =opt_cfg,
                cost_cfg             =self._cost,
                analytical_sla_target=target,
            )
            optimizer._last_sim_feedback = _sim_feedback
            opt_result = optimizer.solve()

            if opt_cfg.verbose:
                print(f"  [Iter {iteration}] CP-SAT status={opt_result.status}")
                for sh, plan in opt_result.shift_plan.items():
                    sh_start = next(s for nm, s, _ in SHIFT_WINDOWS if nm == sh)
                    sh_end   = next(e for nm, _, e in SHIFT_WINDOWS if nm == sh)
                    print(f"    Shift {sh} ({sh_start}-{sh_end}m): {plan}")
                print(f"    Peak coverage: {opt_result.agents_per_skill}"
                      f"  cost=£{opt_result.total_staffing_cost:,.2f}")

            eval_result = self._evaluator.evaluate(opt_result)
            sla_gap     = eval_result.sla - opt_cfg.sla_target
            converged   = sla_gap >= -opt_cfg.convergence_tolerance

            _sim_feedback = {
                "sla_gap":          sla_gap,
                "abandonment_rate": eval_result.abandonment_rate,
            }

            if opt_cfg.verbose:
                print(f"  [Iter {iteration}] Sim SLA={eval_result.sla:.1%}  "
                      f"abandon={eval_result.abandonment_rate:.1%}  "
                      f"gap={sla_gap:+.1%}  "
                      f"{'✓ CONVERGED' if converged else '✗ below target'}")

            loop_iter = _LoopIteration(
                iteration        =iteration,
                analytical_target=target,
                opt_result       =opt_result,
                eval_result      =eval_result,
                sla_gap          =sla_gap,
                converged        =converged,
            )
            self._history.append(loop_iter)

            if converged:
                if opt_cfg.verbose:
                    print(f"\n  ✓ Converged at iteration {iteration}.")
                break

            correction = (
                max(0.01, min(0.10, abs(sla_gap) * 1.5))
                if opt_cfg.analytical_safety_margin is None
                else opt_cfg.analytical_safety_margin
            )
            target = min(0.999, target + correction)

        else:
            if opt_cfg.verbose:
                print(f"\n  ⚠  Max iterations ({opt_cfg.max_iterations}) reached without convergence.")
                best = self._best_history_entry()
                if best:
                    print(f"     Best SLA achieved: {best.eval_result.sla:.1%}")

        if opt_cfg.pareto_sweep:
            self._run_pareto_sweep()

        return self._history

    @property
    def best_plan(self) -> Optional[Dict[str, int]]:
        entry = self._best_history_entry()
        return entry.opt_result.agents_per_skill if entry else None

    @property
    def best_shift_plan(self) -> Optional[Dict[str, Dict[str, int]]]:
        entry = self._best_history_entry()
        return entry.opt_result.shift_plan if entry else None

    @property
    def best_evaluation(self) -> Optional[EvaluationResult]:
        entry = self._best_history_entry()
        return entry.eval_result if entry else None

    def _best_history_entry(self) -> Optional[_LoopIteration]:
        if not self._history:
            return None
        converged = [h for h in self._history if h.converged]
        if converged:
            return converged[-1]
        return max(self._history, key=lambda h: h.eval_result.sla)

    def _run_pareto_sweep(self) -> None:
        sweep = [
            ("cost-only",  1.0, 0.0),
            ("balanced",   0.6, 0.4),
            ("sla-heavy",  0.2, 0.8),
        ]
        self.pareto_points = []
        for label, cw, sw in sweep:
            tmp_cfg             = copy.copy(self._opt)
            tmp_cfg.cost_weight = cw
            tmp_cfg.sla_weight  = sw
            tmp_cfg.verbose     = False
            solver = StaffingOptimizer(
                sim_cfg              =self._base,
                opt_cfg              =tmp_cfg,
                cost_cfg             =self._cost,
                analytical_sla_target=self._opt.sla_target,
            )
            result     = solver.solve()
            avg_erlang = (
                sum(result.analytical_sla.values()) / len(result.analytical_sla)
                if result.analytical_sla else 0.0
            )
            self.pareto_points.append({
                "label":         label,
                "cost_weight":   cw,
                "sla_weight":    sw,
                "peak_plan":     result.agents_per_skill,
                "shift_plan":    result.shift_plan,
                "staffing_cost": result.total_staffing_cost,
                "erlang_sla_avg":avg_erlang,
            })

    def print_pareto_summary(self) -> None:
        if not self.pareto_points:
            print("[pareto] No sweep data — set pareto_sweep=True before run().")
            return
        w = 80
        print(f"\n{'=' * w}")
        print(f"  COST vs SLA TRADE-OFF  (Erlang-C analytical, 720-min horizon)")
        print(f"{'=' * w}")
        print(f"  {'Label':<14}  {'c-wt':>5}  {'s-wt':>5}  "
              f"{'Peak plan':>30}  {'Cost £':>10}  {'SLA (anlyt)':>12}")
        print(f"  {'-' * (w - 4)}")
        for pt in self.pareto_points:
            print(
                f"  {pt['label']:<14}  {pt['cost_weight']:>5.2f}  {pt['sla_weight']:>5.2f}  "
                f"{str(pt['peak_plan']):>30}  £{pt['staffing_cost']:>8,.0f}  "
                f"{pt['erlang_sla_avg']:>12.1%}"
            )
        print(f"{'=' * w}\n")

    def explain_plan(
        self,
        opt_result:  Optional[OptimizationResult] = None,
        eval_result: Optional[EvaluationResult]   = None,
    ) -> None:
        entry = self._best_history_entry()
        opt_result  = opt_result  or (entry.opt_result  if entry else None)
        eval_result = eval_result or (entry.eval_result if entry else None)

        if opt_result is None:
            print("[explain] No plan available — run() first.")
            return

        cfg      = self._base
        opt      = self._opt
        lam      = cfg.arrival_rate_per_minute
        mean_svc = cfg.mean_service_minutes + cfg.acw_mean_minutes
        try:
            wages    = self._cost.hourly_wage_per_skill
            overhead = self._cost.overhead_factor
        except AttributeError:
            wages    = StaffingOptimizer._DEFAULT_WAGES
            overhead = StaffingOptimizer._DEFAULT_OVERHEAD

        w = 72
        print(f"\n{'=' * w}")
        print(f"  STAFFING PLAN EXPLANATION  [OR-Tools CP-SAT, 720-min horizon]")
        print(f"{'=' * w}")
        print(f"  Arrival rate : {cfg.arrival_rate_per_hour:.0f} calls/hr  ({lam:.3f}/min, uniform)")
        print(f"  Service time : {mean_svc:.2f} min (handle + ACW)")
        print(f"  SLA target   : ≥{opt.sla_target:.0%} within {opt.sla_threshold_minutes:.1f} min")
        print(f"  Horizon      : {HORIZON_MINUTES} min")
        if opt.skill_realism_derating:
            print(f"  Realism de-rating : {opt.skill_realism_derating}")
        if opt.realism_floor_agents:
            print(f"  Realism floor     : {opt.realism_floor_agents}")

        if eval_result is not None:
            sla_ok = "✓" if eval_result.sla >= opt.sla_target else "✗"
            print(f"\n  Simulation outcome (720-min run):")
            print(f"    SLA        : {eval_result.sla:.1%}  {sla_ok}")
            print(f"    Abandonment: {eval_result.abandonment_rate:.1%}")
            print(f"    Avg CSAT   : {eval_result.avg_csat:.2f}")
            print(f"    Total cost : £{eval_result.total_cost:,.2f}")

        print(f"\n  Shift plan:")
        print(f"  (util = offered_load / agents_on_this_shift; combined = all shifts active in that band)")
        total_mix_ep = sum(cfg.skill_mix.values()) or 1.0
        for sh_name, sh_start, sh_end in SHIFT_WINDOWS:
            plan = opt_result.shift_plan.get(sh_name, {})
            cost = sum(
                plan.get(skill, 0) * wages.get(skill, 18.0) * overhead
                * StaffingOptimizer._SHIFT_HOURS
                for skill in plan
            )
            break_mid = _SHIFT_BREAK_MIDPOINTS[sh_name]
            print(f"\n  Shift {sh_name}  ({sh_start}\u2013{sh_end}m)  "
                  f"break@{break_mid:.0f}m  shift-cost=\u00a3{cost:,.2f}")
            for skill in sorted(plan.keys()):
                n    = plan[skill]
                frac = cfg.skill_mix.get(skill, 1.0 / max(len(plan), 1)) / total_mix_ep
                # Use the busiest band this shift is active in for the util calculation
                best_lam = max(
                    lam * ((b_end - b_start) / HORIZON_MINUTES) * frac
                    for b_start, b_end, active_shifts in COVERAGE_BANDS
                    if sh_name in active_shifts
                )
                util     = (best_lam * mean_svc) / max(n, 1)
                peak_cov = opt_result.band_coverage.get(PEAK_BAND_INDEX, {}).get(skill, n)
                note     = (
                    f"  (combined: {peak_cov} at peak band)"
                    if n < peak_cov else ""
                )
                print(f"    {skill:<14}  {n} agents  (util\u2248{util:.0%}){note}")

        print(f"\n  Coverage by band:")
        skills = sorted(opt_result.agents_per_skill.keys())
        hdr    = f"  {'Band':<22}  {'Shifts':>10}" + "".join(f"  {s[:7]:>8}" for s in skills)
        print(hdr)
        print(f"  {'-' * (len(hdr) - 2)}")
        for band_idx, (b_start, b_end, active_shifts) in enumerate(COVERAGE_BANDS):
            shift_str = "+".join(active_shifts)
            cov       = opt_result.band_coverage.get(band_idx, {})
            row = (
                f"  Band{band_idx} [{b_start:>4}-{b_end:>4}m]  "
                f"{shift_str:>10}" +
                "".join(f"  {cov.get(s, 0):>8}" for s in skills)
            )
            if band_idx == PEAK_BAND_INDEX:
                row += "  ← peak"
            print(row)

        print(f"\n  Total staffing cost : £{opt_result.total_staffing_cost:,.2f}")
        print(f"{'=' * w}\n")

    def plot_results(self, save_path: Optional[str] = None) -> None:
        if not _MPL_AVAILABLE:
            warnings.warn("matplotlib not installed — cannot plot.", RuntimeWarning)
            return
        if not self._history:
            print("[plot] No history — call run() first.")
            return

        best_ev  = self.best_evaluation
        has_cost = best_ev is not None and bool(best_ev.cost_breakdown)
        n_panels = 3 if has_cost else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
        fig.suptitle(
            f"Staffing Optimiser — {HORIZON_MINUTES}-min Horizon  [OR-Tools CP-SAT]",
            fontweight="bold",
        )

        iters     = [h.iteration              for h in self._history]
        sim_slas  = [h.eval_result.sla * 100  for h in self._history]
        anlyt_tgt = [h.analytical_target * 100 for h in self._history]

        ax1 = axes[0]
        ax1.plot(iters, sim_slas,  "o-",  label="Simulated SLA",    linewidth=2)
        ax1.plot(iters, anlyt_tgt, "s--", label="Analytical target", linewidth=1.5, alpha=0.75)
        ax1.axhline(
            y=self._opt.sla_target * 100, color="red", linestyle=":",
            linewidth=1.2, label=f"SLA target ({self._opt.sla_target:.0%})",
        )
        ax1.set_xlabel("Iteration"); ax1.set_ylabel("SLA (%)"); ax1.set_title("Convergence")
        ax1.set_xticks(iters); ax1.legend(fontsize=8)
        for h in self._history:
            if h.converged:
                ax1.axvline(x=h.iteration, color="green", linestyle="--", linewidth=1.0, alpha=0.7)

        ax2    = axes[1]
        entry  = self._best_history_entry()
        sp     = entry.opt_result.shift_plan if entry else {}
        skills = sorted((entry.opt_result.agents_per_skill or {}).keys())
        x      = np.arange(len(skills))
        width  = 0.25
        colours = ["#4C9BE8", "#56C271", "#A47FE8"]
        for i, (sh_name, _, _) in enumerate(SHIFT_WINDOWS):
            vals = [sp.get(sh_name, {}).get(s, 0) for s in skills]
            bars = ax2.bar(x + (i - 1) * width, vals, width,
                           label=f"Shift {sh_name}", color=colours[i], edgecolor="white")
            for bar, v in zip(bars, vals):
                if v:
                    ax2.text(bar.get_x() + bar.get_width() / 2,
                             bar.get_height() + 0.05, str(v),
                             ha="center", va="bottom", fontsize=8)
        ax2.set_xticks(x); ax2.set_xticklabels(skills)
        ax2.set_xlabel("Skill"); ax2.set_ylabel("Agents")
        ax2.set_title("Multi-Shift Staffing Plan"); ax2.legend(fontsize=8)
        if best_ev:
            ax2.set_xlabel(
                f"SLA {best_ev.sla:.1%}  |  Abandon {best_ev.abandonment_rate:.1%}", fontsize=8,
            )

        if has_cost:
            ax3   = axes[2]
            bk    = {k: v for k, v in best_ev.cost_breakdown.items() if v > 0}
            lbls  = list(bk.keys()); vals = list(bk.values()); total = sum(vals)
            clrs  = ["#4C9BE8", "#F5A623", "#56C271", "#A47FE8", "#F78C6C", "#FFCB6B", "#FF6B6B"]
            ypos  = range(len(lbls))
            ax3.barh(list(ypos), vals, color=clrs[:len(lbls)], edgecolor="white", linewidth=0.6)
            for i, v in enumerate(vals):
                ax3.text(v + total * 0.01, i, f"£{v:,.0f}  ({v/total:.0%})", va="center", fontsize=8)
            ax3.set_yticks(list(ypos))
            ax3.set_yticklabels([lbl.replace("_", " ").title() for lbl in lbls])
            ax3.set_xlabel("Cost (£)"); ax3.set_title(f"Cost Breakdown  (total £{total:,.0f})")
            ax3.invert_yaxis()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  [plot] Saved to {save_path}")
        else:
            plt.show()


# ---------------------------------------------------------------------------
# LoopReport
# ---------------------------------------------------------------------------

class LoopReport:
    def __init__(self, history: List[_LoopIteration], cfg: OptimizationConfig) -> None:
        self._history = history
        self._cfg     = cfg

    def print_report(self) -> None:
        if not self._history:
            print("[LoopReport] No iterations to report.")
            return

        w   = 88
        bar = "=" * w
        cfg = self._cfg
        correction_mode = (
            "adaptive"
            if cfg.analytical_safety_margin is None
            else f"fixed={cfg.analytical_safety_margin:.3f}"
        )

        print(f"\n{bar}")
        print(f"  OPTIMIZE → SIMULATE LOOP  [OR-Tools CP-SAT | {HORIZON_MINUTES}-min horizon]")
        print(f"  Shifts: M(0–480) D(120–600) E(240–720)  |  Peak band: {COVERAGE_BANDS[PEAK_BAND_INDEX][:2]}")
        print(f"  SLA target={cfg.sla_target:.1%}  |  Engine: {cfg.engine_type}"
              f"  |  Correction: {correction_mode}")
        if cfg.sla_violation_penalty_per_call > 0:
            print(f"  SLA-violation penalty : £{cfg.sla_violation_penalty_per_call:.2f}/call")
        if cfg.skill_realism_derating:
            print(f"  Realism de-rating     : {cfg.skill_realism_derating}")
        if cfg.realism_floor_agents:
            print(f"  Realism floor agents  : {cfg.realism_floor_agents}")
        print(f"{bar}")

        skill_list = sorted(self._history[0].opt_result.agents_per_skill.keys())
        pk_hdr     = "  ".join(f"{s[:4]:>5}" for s in skill_list)
        print(
            f"  {'Iter':>4}  {'ATarget':>7}  {pk_hdr}  "
            f"{'SimSLA':>7}  {'Abandon':>8}  {'CSAT':>6}  {'Cost £':>10}  {'OK?':>4}"
        )
        print(f"  {'-' * (w - 2)}")

        for h in self._history:
            plan   = h.opt_result.agents_per_skill
            agents = "  ".join(f"{plan.get(s, 0):>5}" for s in skill_list)
            ev     = h.eval_result
            ok     = "✓" if h.converged else "✗"
            cost_s = f"£{ev.total_cost:>9,.2f}" if ev.total_cost else "    n/a   "
            print(
                f"  {h.iteration:>4}  {h.analytical_target:>7.3%}  {agents}  "
                f"{ev.sla:>7.1%}  {ev.abandonment_rate:>8.1%}  "
                f"{ev.avg_csat:>6.2f}  {cost_s}  {ok:>4}"
            )

        print(f"  {'-' * (w - 2)}")
        best_entry = max(self._history, key=lambda h: h.eval_result.sla)
        best_ev    = best_entry.eval_result
        converged  = any(h.converged for h in self._history)

        print(f"\n  Final status      : {'CONVERGED' if converged else 'MAX ITER REACHED'}")
        print(f"  Iterations run    : {len(self._history)}")
        print(f"  Best SLA achieved : {best_ev.sla:.1%}  (target: {cfg.sla_target:.1%})")
        print(f"  Peak coverage     : {best_ev.agents_per_skill}")

        print(f"\n  Best shift plan:")
        for sh_name, sh_start, sh_end in SHIFT_WINDOWS:
            plan = best_entry.opt_result.shift_plan.get(sh_name, {})
            print(f"    Shift {sh_name} ({sh_start:>4}–{sh_end:>4}m): {plan}")

        if best_ev.total_cost:
            print(f"\n  Total cost        : £{best_ev.total_cost:,.2f}")
            if best_ev.cost_breakdown:
                print(f"\n  Cost breakdown:")
                for component, amount in sorted(
                    best_ev.cost_breakdown.items(), key=lambda x: -x[1]
                ):
                    print(f"    {component:<24}  £{amount:>10,.2f}")

        final_h = self._history[-1]
        print(f"\n  Erlang-C SLA at final plan (per band):")
        for band_idx, (b_start, b_end, active_shifts) in enumerate(COVERAGE_BANDS):
            cov        = final_h.opt_result.band_coverage.get(band_idx, {})
            peak_tag   = "  ← peak" if band_idx == PEAK_BAND_INDEX else ""
            shifts_str = "+".join(active_shifts)
            print(f"    Band{band_idx} [{b_start:>4}–{b_end:>4}m] {shifts_str:<8}{peak_tag}")
            for skill in skill_list:
                n_agents  = cov.get(skill, 0)
                anlyt_sla = final_h.opt_result.analytical_sla.get(skill, 0.0) \
                            if band_idx == PEAK_BAND_INDEX else 0.0
                derating  = (cfg.skill_realism_derating or {}).get(skill, 1.0)
                floor     = (cfg.realism_floor_agents or {}).get(skill, 0)
                notes     = []
                if derating != 1.0 and band_idx == PEAK_BAND_INDEX:
                    notes.append(f"de-rated→{anlyt_sla * derating:.1%}")
                if floor > 0:
                    notes.append(f"+{floor} floor")
                note_str = f"  [{', '.join(notes)}]" if notes else ""
                sla_str  = f"  Erlang-C SLA={anlyt_sla:.1%}" if band_idx == PEAK_BAND_INDEX else ""
                print(f"      {skill:<14}  {n_agents} agents{sla_str}{note_str}")

        print(f"\n{bar}\n")