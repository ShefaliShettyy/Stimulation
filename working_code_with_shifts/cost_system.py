"""
cost_system.py
==============
Business-cost layer for the call-centre simulation.

Purpose
-------
Translate simulation outcomes into money so different staffing plans can be
compared on total cost, not just service level.

Cost components
---------------
- Staffing        : wages * overhead * shift hours per skill (StaffingCostModel).
- Service failure : abandonment penalties, SLA-violation penalties and repeat-
                    call surcharges, each scaled by customer tier
                    (ServiceFailureCostModel).
- Utilisation     : overtime cost above a utilisation threshold and idle cost
                    below another (UtilizationCostModel).
- Burnout         : incident, productivity-drag and attrition costs when an
                    agent's peak fatigue exceeds a threshold (BurnoutCostModel).

How it fits together
--------------------
CostLedger accumulates timestamped cost events during/after a run.
SystemCostFunction turns the ledger plus staffing into a labelled breakdown and
prints a report. CostAwareEngine and CostAwareRealisticsEngine subclass the
core simulation engines and settle utilisation/burnout costs automatically once
a run finishes (CostAwareKPIEngine records per-call failure costs as they
occur). ScenarioCostResult / CostReport compare several scenarios side by side.

Note on costs
-------------
The burnout costs are realistic per-shift *expected* contributions, not
worst-case actuarial values, and the fatigue threshold is tuned for the
720-minute horizon used by the optimiser.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from core_simulation import (
    Agent, Call, KPIEngine, RouterScoreWeights,
    SimulationConfig, SimulationEngine, _CallRecord,
    BehaviorConfig, RealisticsAwareEngine,
)


@dataclass
class CostConfig:
    hourly_wage_per_skill: Dict[str, float] = field(
        default_factory=lambda: {"billing": 18.00, "technical": 22.00, "general": 16.00}
    )
    overhead_factor:  float = 1.30
    shift_hours:      float = 8.0

    abandonment_penalty_base: float = 25.00
    abandonment_churn_factor: float = 80.00
    sla_violation_penalty:    float = 8.00
    sla_threshold_minutes:    float = 1.0
    repeat_call_surcharge:    float = 6.00

    tier_multipliers: Dict[str, float] = field(
        default_factory=lambda: {"vip": 3.0, "premium": 1.5, "standard": 1.0}
    )
    churn_probability: Dict[str, float] = field(
        default_factory=lambda: {"vip": 0.40, "premium": 0.20, "standard": 0.05}
    )

    overtime_utilization_threshold: float = 0.85
    overtime_rate_multiplier:       float = 1.50
    idle_utilization_threshold:     float = 0.40
    idle_cost_rate:                 float = 0.60

    # Per-shift burnout costs (expected-value, not worst-case); threshold tuned
    # for the 720-minute horizon where agents accumulate more fatigue.
    burnout_fatigue_threshold: float = 0.80
    burnout_incident_cost:     float = 500.00
    burnout_productivity_drag: float = 5.00
    burnout_attrition_prob:    float = 0.02
    burnout_replacement_cost:  float = 750.00


class StaffingCostModel:
    def __init__(self, sim_cfg: SimulationConfig, cost_cfg: CostConfig) -> None:
        self._sim  = sim_cfg
        self._cost = cost_cfg

    def cost_per_skill(self) -> Dict[str, float]:
        return {
            skill: round(
                n * self._cost.hourly_wage_per_skill.get(skill, 18.0)
                  * self._cost.overhead_factor * self._cost.shift_hours, 2,
            )
            for skill, n in self._sim.agents_per_skill.items()
        }

    def total_staffing_cost(self) -> float:
        return sum(self.cost_per_skill().values())

    def cost_per_call(self, total_calls: int) -> float:
        return 0.0 if total_calls == 0 else self.total_staffing_cost() / total_calls


class ServiceFailureCostModel:
    def __init__(self, cost_cfg: CostConfig) -> None:
        self._cfg = cost_cfg

    def abandonment_cost(self, customer_type: str) -> float:
        cfg     = self._cfg
        mult    = cfg.tier_multipliers.get(customer_type, 1.0)
        churn_p = cfg.churn_probability.get(customer_type, 0.05)
        return round(cfg.abandonment_penalty_base * mult + churn_p * cfg.abandonment_churn_factor * mult, 4)

    def sla_violation_cost(self, customer_type: str) -> float:
        mult = self._cfg.tier_multipliers.get(customer_type, 1.0)
        return round(self._cfg.sla_violation_penalty * mult, 4)

    def repeat_call_cost(self) -> float:
        return self._cfg.repeat_call_surcharge


@dataclass
class AgentUtilizationStats:
    agent_id:         str
    primary_skill:    str
    total_handle_min: float
    shift_minutes:    float
    peak_fatigue:     float = 0.0
    calls_handled:    int   = 0

    @property
    def utilization(self) -> float:
        if self.shift_minutes <= 0:
            return 0.0
        return min(1.0, self.total_handle_min / self.shift_minutes)

    def __repr__(self) -> str:
        return (
            f"AgentUtilizationStats({self.agent_id}, util={self.utilization:.2%}, "
            f"fatigue={self.peak_fatigue:.3f}, calls={self.calls_handled})"
        )


class UtilizationCostModel:
    def __init__(self, cost_cfg: CostConfig, sim_cfg: SimulationConfig) -> None:
        self._cost = cost_cfg
        self._sim  = sim_cfg

    def overtime_cost_for_agent(self, stats: AgentUtilizationStats) -> float:
        util = stats.utilization
        cfg  = self._cost
        if util <= cfg.overtime_utilization_threshold:
            return 0.0
        excess_frac    = util - cfg.overtime_utilization_threshold
        overtime_hours = excess_frac * cfg.shift_hours
        wage           = cfg.hourly_wage_per_skill.get(stats.primary_skill, 18.0)
        return round(overtime_hours * wage * cfg.overhead_factor * cfg.overtime_rate_multiplier, 4)

    def idle_cost_for_agent(self, stats: AgentUtilizationStats) -> float:
        util = stats.utilization
        cfg  = self._cost
        if util >= cfg.idle_utilization_threshold:
            return 0.0
        idle_hours = (cfg.idle_utilization_threshold - util) * cfg.shift_hours
        wage       = cfg.hourly_wage_per_skill.get(stats.primary_skill, 18.0)
        return round(idle_hours * wage * cfg.idle_cost_rate, 4)

    def compute_all(
        self, all_stats: List[AgentUtilizationStats]
    ) -> Tuple[float, float, Dict]:
        total_ot = total_idle = 0.0
        per_agent: Dict[str, Dict[str, float]] = {}
        for st in all_stats:
            ot   = self.overtime_cost_for_agent(st)
            idle = self.idle_cost_for_agent(st)
            total_ot   += ot
            total_idle += idle
            per_agent[st.agent_id] = {
                "overtime":    round(ot,   4),
                "idle":        round(idle, 4),
                "utilization": round(st.utilization, 4),
            }
        return round(total_ot, 4), round(total_idle, 4), per_agent


class BurnoutCostModel:
    def __init__(self, cost_cfg: CostConfig) -> None:
        self._cfg = cost_cfg

    def burnout_cost_for_agent(self, stats: AgentUtilizationStats) -> Dict[str, float]:
        cfg    = self._cfg
        excess = stats.peak_fatigue - cfg.burnout_fatigue_threshold
        if excess <= 0:
            return {"incident": 0.0, "productivity_drag": 0.0, "attrition_risk": 0.0, "total": 0.0}
        headroom          = max(1e-9, 1.0 - cfg.burnout_fatigue_threshold)
        severity          = (excess / headroom) ** 2
        incident          = cfg.burnout_incident_cost     * severity
        productivity_drag = cfg.burnout_productivity_drag * severity
        attrition_risk    = cfg.burnout_attrition_prob * cfg.burnout_replacement_cost * severity
        total             = incident + productivity_drag + attrition_risk
        return {
            "incident":          round(incident,          4),
            "productivity_drag": round(productivity_drag, 4),
            "attrition_risk":    round(attrition_risk,    4),
            "total":             round(total,             4),
        }

    def compute_all(
        self, all_stats: List[AgentUtilizationStats]
    ) -> Tuple[float, Dict]:
        total = 0.0
        per_agent: Dict[str, Dict[str, float]] = {}
        for st in all_stats:
            bd = self.burnout_cost_for_agent(st)
            total += bd["total"]
            per_agent[st.agent_id] = bd
        return round(total, 4), per_agent


class AgentUtilizationCollector:
    def __init__(self, engine) -> None:
        self._engine = engine

    def collect(self) -> List[AgentUtilizationStats]:
        cfg     = self._engine.config
        agents  = self._engine.agents
        realism = getattr(self._engine, "realism", None)
        result  = []
        for agent in agents:
            router = getattr(self._engine, "router", None)
            trk_st = router.tracker.get(agent.agent_id) if router else None
            if realism is not None and agent.agent_id in realism._states:
                real_st          = realism._states[agent.agent_id]
                total_handle_min = real_st.total_handle_minutes
                calls_handled    = real_st.calls_handled
                peak_fatigue     = real_st.peak_fatigue_level
            else:
                total_handle_min = getattr(trk_st, "total_handle", 0.0)
                calls_handled    = getattr(trk_st, "calls_completed", 0)
                peak_fatigue     = 0.0
            result.append(AgentUtilizationStats(
                agent_id        =agent.agent_id,
                primary_skill   =agent.primary_skill,
                total_handle_min=total_handle_min,
                shift_minutes   =cfg.sim_duration_minutes,
                peak_fatigue    =peak_fatigue,
                calls_handled   =calls_handled,
            ))
        return result


@dataclass
class _CostEvent:
    sim_time:      float
    event_type:    str
    customer_type: str
    skill:         str
    amount:        float
    agent_id:      Optional[str] = None


class CostLedger:
    def __init__(self, cost_cfg: CostConfig) -> None:
        self._failure_model = ServiceFailureCostModel(cost_cfg)
        self._events: List[_CostEvent] = []

    def record_abandonment(self, call: Call, sim_time: float) -> float:
        cost = self._failure_model.abandonment_cost(call.customer_type)
        self._events.append(_CostEvent(
            sim_time=sim_time, event_type="abandonment",
            customer_type=call.customer_type, skill=call.skill, amount=cost,
        ))
        return cost

    def record_answered_call(self, record: _CallRecord) -> float:
        total = 0.0
        if record.wait_minutes > self._failure_model._cfg.sla_threshold_minutes:
            cost = self._failure_model.sla_violation_cost(record.customer_type)
            self._events.append(_CostEvent(
                sim_time=record.service_start, event_type="sla_violation",
                customer_type=record.customer_type, skill=record.skill, amount=cost,
            ))
            total += cost
        if record.is_repeat:
            cost = self._failure_model.repeat_call_cost()
            self._events.append(_CostEvent(
                sim_time=record.service_start, event_type="repeat_call",
                customer_type=record.customer_type, skill=record.skill, amount=cost,
            ))
            total += cost
        return total

    def record_staffing_cost(self, skill: str, amount: float) -> None:
        self._events.append(_CostEvent(
            sim_time=0.0, event_type="staffing",
            customer_type="n/a", skill=skill, amount=amount,
        ))

    def record_utilization_costs(
        self, per_agent: Dict, agent_skill_map: Dict[str, str], sim_time: float
    ) -> Tuple[float, float]:
        total_ot = total_idle = 0.0
        for agent_id, costs in per_agent.items():
            skill = agent_skill_map.get(agent_id, "general")
            if costs["overtime"] > 0.0:
                self._events.append(_CostEvent(
                    sim_time=sim_time, event_type="overtime",
                    customer_type="n/a", skill=skill, amount=costs["overtime"], agent_id=agent_id,
                ))
                total_ot += costs["overtime"]
            if costs["idle"] > 0.0:
                self._events.append(_CostEvent(
                    sim_time=sim_time, event_type="idle",
                    customer_type="n/a", skill=skill, amount=costs["idle"], agent_id=agent_id,
                ))
                total_idle += costs["idle"]
        return round(total_ot, 4), round(total_idle, 4)

    def record_burnout_costs(
        self, per_agent: Dict, agent_skill_map: Dict[str, str], sim_time: float
    ) -> float:
        total = 0.0
        for agent_id, costs in per_agent.items():
            if costs["total"] <= 0.0:
                continue
            self._events.append(_CostEvent(
                sim_time=sim_time, event_type="burnout",
                customer_type="n/a", skill=agent_skill_map.get(agent_id, "general"),
                amount=costs["total"], agent_id=agent_id,
            ))
            total += costs["total"]
        return round(total, 4)

    def total_by_type(self) -> Dict[str, float]:
        r: Dict[str, float] = {}
        for ev in self._events:
            r[ev.event_type] = r.get(ev.event_type, 0.0) + ev.amount
        return r

    def total_by_skill(self) -> Dict[str, float]:
        r: Dict[str, float] = {}
        for ev in self._events:
            r[ev.skill] = r.get(ev.skill, 0.0) + ev.amount
        return r

    def total_by_agent(self) -> Dict[str, float]:
        r: Dict[str, float] = {}
        for ev in self._events:
            if ev.agent_id:
                r[ev.agent_id] = r.get(ev.agent_id, 0.0) + ev.amount
        return r

    def total_cost(self) -> float:
        return sum(ev.amount for ev in self._events)


class SystemCostFunction:
    def __init__(
        self,
        staffing_model: StaffingCostModel,
        ledger:         CostLedger,
        sim_cfg:        SimulationConfig,
    ) -> None:
        self._staffing = staffing_model
        self._ledger   = ledger
        self._sim      = sim_cfg

    def breakdown(self) -> Dict[str, float]:
        by_type = self._ledger.total_by_type()
        result  = {
            "staffing":      self._staffing.total_staffing_cost(),
            "abandonment":   by_type.get("abandonment",   0.0),
            "sla_violation": by_type.get("sla_violation", 0.0),
            "repeat_call":   by_type.get("repeat_call",   0.0),
            "overtime":      by_type.get("overtime",      0.0),
            "idle":          by_type.get("idle",          0.0),
            "burnout":       by_type.get("burnout",       0.0),
        }
        result["total"] = sum(result.values())
        return result

    def total(self) -> float:
        return self.breakdown()["total"]

    def cost_per_handled_call(self, total_calls: int) -> float:
        return 0.0 if total_calls == 0 else self.total() / total_calls

    def report(self, label: str = "Scenario") -> None:
        w   = 66
        bar = "=" * w
        bk  = self.breakdown()
        print(f"\n{bar}")
        print(f"  BUSINESS COST REPORT  --  {label}")
        print(f"{bar}")
        components = [
            ("Staffing cost",         bk["staffing"]),
            ("Abandonment penalty",   bk["abandonment"]),
            ("SLA violation penalty", bk["sla_violation"]),
            ("Repeat call surcharge", bk["repeat_call"]),
            ("Overtime cost",         bk["overtime"]),
            ("Idle cost",             bk["idle"]),
            ("Burnout cost",          bk["burnout"]),
        ]
        for lbl, amount in components:
            pct     = (amount / bk["total"] * 100) if bk["total"] else 0
            bar_len = int(pct / 2)
            print(f"  {lbl:<28}  £{amount:>10,.2f}  {pct:>5.1f}%  {'#' * bar_len}")
        print(f"  {'-' * (w - 4)}")
        print(f"  {'TOTAL SHIFT COST':<28}  £{bk['total']:>10,.2f}")
        print(f"{bar}")


@dataclass
class ScenarioCostResult:
    label:              str
    staffing:           float
    abandonment:        float
    sla_violation:      float
    repeat_call:        float
    overtime:           float
    idle:               float
    burnout:            float
    total:              float
    total_calls:        int
    total_abandonments: int

    @classmethod
    def from_engine(cls, label: str, engine) -> "ScenarioCostResult":
        bk = engine.cost_function.breakdown()
        return cls(
            label=label, staffing=bk["staffing"], abandonment=bk["abandonment"],
            sla_violation=bk["sla_violation"], repeat_call=bk["repeat_call"],
            overtime=bk.get("overtime", 0.0), idle=bk.get("idle", 0.0),
            burnout=bk.get("burnout", 0.0), total=bk["total"],
            total_calls=engine.kpi.total_calls(),
            total_abandonments=engine.kpi.total_abandonments(),
        )


class CostReport:
    def __init__(self, results: List[ScenarioCostResult]) -> None:
        self._results = results

    def print_report(self) -> None:
        labels = [r.label for r in self._results]
        w      = 28 + 14 * len(labels)
        bar    = "=" * w
        print(f"\n{bar}")
        print(f"  MULTI-SCENARIO COST COMPARISON")
        print(f"{bar}")
        header = f"  {'Component':<28}" + "".join(f"{lbl:>14}" for lbl in labels)
        print(header)
        print(f"  {'-' * (w - 4)}")

        def row(name: str, getter):
            vals = "".join(f"  £{getter(r):>10,.2f}" for r in self._results)
            print(f"  {name:<28}{vals}")

        row("Staffing cost",         lambda r: r.staffing)
        row("Abandonment penalty",   lambda r: r.abandonment)
        row("SLA violation penalty", lambda r: r.sla_violation)
        row("Repeat call surcharge", lambda r: r.repeat_call)
        row("Overtime cost",         lambda r: r.overtime)
        row("Idle cost",             lambda r: r.idle)
        row("Burnout cost",          lambda r: r.burnout)
        print(f"  {'-' * (w - 4)}")
        row("TOTAL COST",            lambda r: r.total)
        print(f"{bar}\n")


# ---------------------------------------------------------------------------
# CostAwareKPIEngine
# ---------------------------------------------------------------------------

class CostAwareKPIEngine(KPIEngine):
    def __init__(self, ledger: CostLedger, cost_cfg: CostConfig) -> None:
        super().__init__()
        self._ledger   = ledger
        self._cost_cfg = cost_cfg

    def record_call(self, record: _CallRecord) -> None:
        super().record_call(record)
        self._ledger.record_answered_call(record)

    def record_abandonment(self, call: Call, abandon_time: float) -> None:
        super().record_abandonment(call, abandon_time)
        self._ledger.record_abandonment(call, abandon_time)


# ---------------------------------------------------------------------------
# CostAwareEngine
# ---------------------------------------------------------------------------

class CostAwareEngine(SimulationEngine):
    def __init__(
        self,
        config:   SimulationConfig,
        cost_cfg: Optional[CostConfig] = None,
        weights:  Optional[RouterScoreWeights] = None,
    ) -> None:
        super().__init__(config, weights)
        cost_cfg             = cost_cfg or CostConfig()
        self._cost_cfg       = cost_cfg
        self._staffing_model = StaffingCostModel(config, cost_cfg)
        self.ledger          = CostLedger(cost_cfg)
        self.cost_function   = SystemCostFunction(self._staffing_model, self.ledger, config)
        self._util_model     = UtilizationCostModel(cost_cfg, config)
        self._burnout_model  = BurnoutCostModel(cost_cfg)
        self.kpi             = CostAwareKPIEngine(self.ledger, cost_cfg)

    def run(self) -> None:
        for skill, cost in self._staffing_model.cost_per_skill().items():
            self.ledger.record_staffing_cost(skill, cost)
        super().run()
        self._settle_post_run_costs()

    def _settle_post_run_costs(self) -> None:
        all_stats       = AgentUtilizationCollector(self).collect()
        agent_skill_map = {a.agent_id: a.primary_skill for a in self.agents}
        sim_time        = self.config.sim_duration_minutes
        _, _, util_pa   = self._util_model.compute_all(all_stats)
        self.ledger.record_utilization_costs(util_pa, agent_skill_map, sim_time)
        _, burnout_pa   = self._burnout_model.compute_all(all_stats)
        self.ledger.record_burnout_costs(burnout_pa, agent_skill_map, sim_time)


# ---------------------------------------------------------------------------
# CostAwareRealisticsEngine
# ---------------------------------------------------------------------------

class CostAwareRealisticsEngine(RealisticsAwareEngine):
    def __init__(
        self,
        config:   SimulationConfig,
        cost_cfg: Optional[CostConfig] = None,
        behavior: Optional[BehaviorConfig] = None,
        weights:  Optional[RouterScoreWeights] = None,
    ) -> None:
        super().__init__(config, behavior, weights)
        cost_cfg             = cost_cfg or CostConfig()
        self._cost_cfg       = cost_cfg
        self._staffing_model = StaffingCostModel(config, cost_cfg)
        self.ledger          = CostLedger(cost_cfg)
        self.cost_function   = SystemCostFunction(self._staffing_model, self.ledger, config)
        self._util_model     = UtilizationCostModel(cost_cfg, config)
        self._burnout_model  = BurnoutCostModel(cost_cfg)
        self.kpi             = CostAwareKPIEngine(self.ledger, cost_cfg)

    def run(self) -> None:
        for skill, cost in self._staffing_model.cost_per_skill().items():
            self.ledger.record_staffing_cost(skill, cost)
        super().run()
        self._settle_post_run_costs()

    def _settle_post_run_costs(self) -> None:
        all_stats       = AgentUtilizationCollector(self).collect()
        agent_skill_map = {a.agent_id: a.primary_skill for a in self.agents}
        sim_time        = self.config.sim_duration_minutes
        _, _, util_pa   = self._util_model.compute_all(all_stats)
        self.ledger.record_utilization_costs(util_pa, agent_skill_map, sim_time)
        _, burnout_pa   = self._burnout_model.compute_all(all_stats)
        self.ledger.record_burnout_costs(burnout_pa, agent_skill_map, sim_time)