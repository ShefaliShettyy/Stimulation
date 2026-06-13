"""
core_simulation.py  (FIXED)
===========================
Fixes applied:
  1. Per-agent CSAT bias stored on Agent so ML affinity scores diverge meaningfully
  2. Agent.csat_bias drawn from N(0, 0.5) — large enough spread to differentiate agents
  3. _sample_csat now accepts optional agent_bias parameter
  4. _pick_agent_id returns Agent object (not just id) so bias flows through
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional, Tuple

import simpy


# ---------------------------------------------------------------------------
# SimulationConfig
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    sim_duration_minutes:    float = 480.0
    arrival_rate_per_hour:   float = 120.0
    skill_mix: Dict[str, float] = field(
        default_factory=lambda: {"billing": 0.40, "technical": 0.35, "general": 0.25}
    )
    customer_tier_mix: Dict[str, float] = field(
        default_factory=lambda: {"vip": 0.10, "premium": 0.25, "standard": 0.65}
    )
    repeat_call_probability: float = 0.12
    mean_service_minutes:    float = 5.0
    stdev_service_minutes:   float = 2.0
    acw_mean_minutes:        float = 1.5
    acw_stdev_minutes:       float = 0.5
    agents_per_skill: Dict[str, int] = field(
        default_factory=lambda: {"billing": 4, "technical": 4, "general": 3}
    )
    overflow_threshold: int = 5
    break_schedule: List[Tuple[float, float]] = field(
        default_factory=lambda: [(120.0, 15.0), (300.0, 30.0), (420.0, 15.0)]
    )
    random_seed: Optional[int] = 42

    @property
    def arrival_rate_per_minute(self) -> float:
        return self.arrival_rate_per_hour / 60.0


# ---------------------------------------------------------------------------
# Call
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Call:
    call_id:       str
    skill:         str
    customer_type: str
    is_repeat:     bool
    arrival_time:  float

    @property
    def priority(self) -> int:
        return {"vip": 0, "premium": 1, "standard": 2}.get(self.customer_type, 2)


# ---------------------------------------------------------------------------
# Agent  — FIX 1: add csat_bias so agents differ meaningfully
# ---------------------------------------------------------------------------

@dataclass
class Agent:
    agent_id:         str
    primary_skill:    str
    secondary_skills: List[str] = field(default_factory=list)
    experience:       str = "mid"
    # Per-agent CSAT bias: drawn from N(0, 0.5) in _create_agent_pool.
    # This makes some agents genuinely better/worse so ML affinity learns real signal.
    csat_bias:        float = field(default=0.0)
    on_break:         bool  = field(default=False, init=False)

    def __repr__(self) -> str:
        status = "break" if self.on_break else "active"
        return f"Agent({self.agent_id}, skill={self.primary_skill}, exp={self.experience}, bias={self.csat_bias:+.2f}, {status})"


# ---------------------------------------------------------------------------
# AgentPerformanceStats
# ---------------------------------------------------------------------------

@dataclass
class AgentPerformanceStats:
    agent_id:        str
    ema_csat:        float = 0.5
    ema_handle_time: float = 5.0
    ema_resolution:  float = 0.85
    calls_completed: int   = 0
    total_csat:      float = 0.0
    total_handle:    float = 0.0
    active_calls:    int   = 0

    EMA_ALPHA:          ClassVar[float] = 0.25
    MIN_RELIABLE_CALLS: ClassVar[int]   = 5

    def record_call(self, csat_raw: float, handle_minutes: float, is_repeat: bool) -> None:
        csat_norm  = (csat_raw - 1.0) / 4.0
        resolution = 0.0 if is_repeat else 1.0
        if self.calls_completed == 0:
            self.ema_csat        = csat_norm
            self.ema_handle_time = handle_minutes
            self.ema_resolution  = resolution
        else:
            a = self.EMA_ALPHA
            self.ema_csat        = a * csat_norm      + (1 - a) * self.ema_csat
            self.ema_handle_time = a * handle_minutes + (1 - a) * self.ema_handle_time
            self.ema_resolution  = a * resolution     + (1 - a) * self.ema_resolution
        self.calls_completed += 1
        self.total_csat      += csat_raw
        self.total_handle    += handle_minutes

    def reliability_weight(self) -> float:
        if self.calls_completed >= self.MIN_RELIABLE_CALLS:
            return 1.0
        return 0.5 + 0.5 * (self.calls_completed / self.MIN_RELIABLE_CALLS)

    def avg_csat_raw(self) -> float:
        if self.calls_completed == 0:
            return 3.5
        return self.total_csat / self.calls_completed

    def __repr__(self) -> str:
        return (
            f"AgentPerformanceStats({self.agent_id}, "
            f"calls={self.calls_completed}, "
            f"ema_csat={self.ema_csat:.3f}, "
            f"ema_hdl={self.ema_handle_time:.2f}m, "
            f"ema_res={self.ema_resolution:.3f})"
        )


# ---------------------------------------------------------------------------
# AgentPerformanceTracker
# ---------------------------------------------------------------------------

class AgentPerformanceTracker:
    def __init__(self, agent_pool: List[Agent], config: SimulationConfig) -> None:
        default_handle = config.mean_service_minutes + config.acw_mean_minutes
        self._stats: Dict[str, AgentPerformanceStats] = {
            a.agent_id: AgentPerformanceStats(
                agent_id=a.agent_id,
                ema_csat=0.5,
                ema_handle_time=default_handle,
                ema_resolution=0.85,
            )
            for a in agent_pool
        }

    def get(self, agent_id: str) -> AgentPerformanceStats:
        return self._stats[agent_id]

    def record_call_start(self, agent_id: str) -> None:
        if agent_id in self._stats:
            self._stats[agent_id].active_calls += 1

    def record_call_end(self, agent_id: str, csat_raw: float, handle_minutes: float, is_repeat: bool) -> None:
        if agent_id not in self._stats:
            return
        st = self._stats[agent_id]
        st.active_calls = max(0, st.active_calls - 1)
        st.record_call(csat_raw, handle_minutes, is_repeat)

    def all_stats(self) -> List[AgentPerformanceStats]:
        return sorted(self._stats.values(), key=lambda s: s.agent_id)

    def print_summary(self) -> None:
        print("\n-- Agent Performance Tracker ---------------------------------------")
        print(f"  {'Agent':<20}  {'Calls':>6}  {'AvgCSAT':>8}  {'EMA Hdl':>8}  {'EMA Res':>8}  {'Active':>7}")
        print("  " + "-" * 68)
        for st in self.all_stats():
            print(
                f"  {st.agent_id:<20}  {st.calls_completed:>6}  "
                f"{st.avg_csat_raw():>8.3f}  {st.ema_handle_time:>7.2f}m  "
                f"{st.ema_resolution:>8.3f}  {st.active_calls:>7}"
            )
        print("-" * 72)


# ---------------------------------------------------------------------------
# RouterScoreWeights
# ---------------------------------------------------------------------------

@dataclass
class RouterScoreWeights:
    performance: float = 0.35
    efficiency:  float = 0.20
    resolution:  float = 0.20
    workload:    float = 0.10
    wait_time:   float = 0.10
    vip_fit:     float = 0.05

    def as_vector(self) -> Tuple[float, ...]:
        return (self.performance, self.efficiency, self.resolution,
                self.workload, self.wait_time, self.vip_fit)

    def normalised(self) -> "RouterScoreWeights":
        total = sum(self.as_vector()) or 1.0
        return RouterScoreWeights(
            performance=self.performance / total,
            efficiency =self.efficiency  / total,
            resolution =self.resolution  / total,
            workload   =self.workload    / total,
            wait_time  =self.wait_time   / total,
            vip_fit    =self.vip_fit     / total,
        )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class Router:
    _MAX_HANDLE_MINUTES: ClassVar[float] = 15.0
    _MAX_ACTIVE_CALLS:   ClassVar[int]   = 5
    _MAX_WAIT_MINUTES:   ClassVar[float] = 10.0
    _VIP_EXPERIENCE_SCORES: ClassVar[Dict[str, float]] = {
        "senior": 1.0, "mid": 0.6, "junior": 0.2,
    }

    def __init__(
        self,
        agent_pool: List[Agent],
        config:     SimulationConfig,
        weights:    Optional[RouterScoreWeights] = None,
    ) -> None:
        self.agent_pool = agent_pool
        self.config     = config
        self.weights    = (weights or RouterScoreWeights()).normalised()
        self.tracker    = AgentPerformanceTracker(agent_pool, config)
        self._agents_by_skill: Dict[str, List[Agent]] = {}
        for agent in agent_pool:
            self._agents_by_skill.setdefault(agent.primary_skill, []).append(agent)

    def select_resource(
        self,
        call:            Call,
        skill_resources: Dict[str, simpy.PriorityResource],
    ) -> Tuple[simpy.PriorityResource, str]:
        skill            = call.skill
        primary_resource = skill_resources[skill]

        if call.customer_type == "vip":
            best_agent, best_score = self._best_agent_for_skill(skill, call, skill_resources)
            if best_agent is not None:
                return primary_resource, f"vip_direct->{best_agent.agent_id}(score={best_score:.3f})"

        q_depth = len(primary_resource.queue)
        if q_depth >= self.config.overflow_threshold:
            overflow_resource, overflow_skill = self._find_overflow_resource(skill, skill_resources)
            if overflow_resource is not None:
                return overflow_resource, f"overflow(q={q_depth})->{overflow_skill}"

        best_agent, best_score = self._best_agent_for_skill(skill, call, skill_resources)
        if best_agent is not None:
            return primary_resource, f"scored->{best_agent.agent_id}(score={best_score:.3f})"

        return primary_resource, "fallback(pool)"

    def estimated_wait(self, resource: simpy.PriorityResource) -> float:
        q = len(resource.queue)
        if q == 0:
            return 0.0
        avg_handle = self._avg_handle_time_for_resource(resource)
        return max(0.0, (q / max(resource.capacity, 1)) * avg_handle)

    def notify_call_started(self, agent_id: str) -> None:
        self.tracker.record_call_start(agent_id)

    def notify_call_ended(
        self,
        agent_id:       str,
        csat_raw:       float,
        handle_minutes: float,
        is_repeat:      bool,
        skill:          str = "",
    ) -> None:
        self.tracker.record_call_end(agent_id, csat_raw, handle_minutes, is_repeat)

    def score_breakdown(
        self,
        agent:           Agent,
        call:            Call,
        skill_resources: Dict[str, simpy.PriorityResource],
    ) -> str:
        signals     = self._compute_signals(agent, call, skill_resources)
        w           = self.weights
        raw_score   = sum(getattr(w, k) * v for k, v in signals.items())
        reliability = self.tracker.get(agent.agent_id).reliability_weight()
        final_score = reliability * raw_score + (1.0 - reliability) * 0.5
        lines = [
            f"Score breakdown | agent={agent.agent_id} | call={call.call_id} | tier={call.customer_type}",
            f"  {'Factor':<14}  {'Weight':>7}  {'Signal':>7}  {'Contrib':>8}",
            "  " + "-" * 44,
        ]
        for name, sig in signals.items():
            wt = getattr(w, name)
            lines.append(f"  {name:<14}  {wt:>7.3f}  {sig:>7.3f}  {wt * sig:>8.4f}")
        lines += [
            "  " + "-" * 44,
            f"  Raw score           {raw_score:>7.4f}",
            f"  Reliability weight  {reliability:>7.4f}",
            f"  Final score         {final_score:>7.4f}",
        ]
        return "\n".join(lines)

    def _best_agent_for_skill(
        self,
        skill:           str,
        call:            Call,
        skill_resources: Dict[str, simpy.PriorityResource],
    ) -> Tuple[Optional[Agent], float]:
        candidates = [a for a in self._agents_by_skill.get(skill, []) if not a.on_break]
        if not candidates:
            return None, 0.0
        best_agent: Optional[Agent] = None
        best_score = -1.0
        for agent in candidates:
            score = self._composite_score(agent, call, skill_resources)
            if score > best_score:
                best_score = score
                best_agent = agent
        return best_agent, best_score

    def _composite_score(
        self,
        agent:           Agent,
        call:            Call,
        skill_resources: Dict[str, simpy.PriorityResource],
    ) -> float:
        signals     = self._compute_signals(agent, call, skill_resources)
        w           = self.weights
        raw_score   = sum(getattr(w, k) * v for k, v in signals.items())
        reliability = self.tracker.get(agent.agent_id).reliability_weight()
        return reliability * raw_score + (1.0 - reliability) * 0.5

    def _compute_signals(
        self,
        agent:           Agent,
        call:            Call,
        skill_resources: Dict[str, simpy.PriorityResource],
    ) -> Dict[str, float]:
        st = self.tracker.get(agent.agent_id)
        performance_sig = float(max(0.0, min(1.0, st.ema_csat)))
        efficiency_sig  = float(max(0.0, min(1.0, 1.0 - st.ema_handle_time / self._MAX_HANDLE_MINUTES)))
        resolution_sig  = float(max(0.0, min(1.0, st.ema_resolution)))
        workload_sig    = float(max(0.0, min(1.0, 1.0 - st.active_calls / self._MAX_ACTIVE_CALLS)))
        resource        = skill_resources.get(agent.primary_skill)
        pred_wait       = self.estimated_wait(resource) if resource else self._MAX_WAIT_MINUTES
        wait_time_sig   = float(max(0.0, min(1.0, 1.0 - pred_wait / self._MAX_WAIT_MINUTES)))
        vip_fit_sig     = (
            self._VIP_EXPERIENCE_SCORES.get(agent.experience, 0.5)
            if call.customer_type == "vip" else 0.5
        )
        return {
            "performance": performance_sig,
            "efficiency":  efficiency_sig,
            "resolution":  resolution_sig,
            "workload":    workload_sig,
            "wait_time":   wait_time_sig,
            "vip_fit":     float(vip_fit_sig),
        }

    def _find_overflow_resource(
        self,
        skill:           str,
        skill_resources: Dict[str, simpy.PriorityResource],
    ) -> Tuple[Optional[simpy.PriorityResource], Optional[str]]:
        cross_options: Dict[str, int] = {}
        for agent in self.agent_pool:
            if skill in agent.secondary_skills and not agent.on_break:
                alt_skill = agent.primary_skill
                if alt_skill != skill and alt_skill in skill_resources:
                    depth = len(skill_resources[alt_skill].queue)
                    if alt_skill not in cross_options or depth < cross_options[alt_skill]:
                        cross_options[alt_skill] = depth
        if not cross_options:
            return None, None
        best_skill = min(cross_options, key=lambda s: cross_options[s])
        return skill_resources[best_skill], best_skill

    def _avg_handle_time_for_resource(self, resource: simpy.PriorityResource) -> float:
        handle_times = [
            st.ema_handle_time
            for agent in self.agent_pool
            for st in [self.tracker.get(agent.agent_id)]
            if st.calls_completed > 0
        ]
        if handle_times:
            return sum(handle_times) / len(handle_times)
        return self.config.mean_service_minutes + self.config.acw_mean_minutes


# ---------------------------------------------------------------------------
# _CallRecord + KPIEngine
# ---------------------------------------------------------------------------

@dataclass
class _CallRecord:
    call_id:        str
    skill:          str
    customer_type:  str
    is_repeat:      bool
    arrival_time:   float
    service_start:  float
    service_end:    float
    csat_raw:       float
    routing_reason: str
    abandoned:      bool = False

    @property
    def wait_minutes(self) -> float:
        return max(0.0, self.service_start - self.arrival_time)

    @property
    def handle_minutes(self) -> float:
        return max(0.0, self.service_end - self.service_start)


class KPIEngine:
    SLA_THRESHOLD_MINUTES: ClassVar[float] = 1.0

    def __init__(self) -> None:
        self._records:      List[_CallRecord] = []
        self._abandonments: List[Dict]        = []

    def record_call(self, record: _CallRecord) -> None:
        self._records.append(record)

    def record_abandonment(self, call: Call, abandon_time: float) -> None:
        self._abandonments.append({
            "call_id":       call.call_id,
            "skill":         call.skill,
            "customer_type": call.customer_type,
            "abandon_time":  abandon_time,
            "arrival_time":  call.arrival_time,
        })

    def total_calls(self) -> int:
        return len(self._records)

    def total_abandonments(self) -> int:
        return len(self._abandonments)

    def abandonment_rate(self) -> float:
        total = self.total_calls() + self.total_abandonments()
        return self.total_abandonments() / total if total else 0.0

    def average_speed_of_answer(self) -> float:
        if not self._records:
            return 0.0
        return sum(r.wait_minutes for r in self._records) / len(self._records)

    def average_handle_time(self) -> float:
        if not self._records:
            return 0.0
        return sum(r.handle_minutes for r in self._records) / len(self._records)

    def sla_percentage(self) -> float:
        if not self._records:
            return 0.0
        within = sum(1 for r in self._records if r.wait_minutes <= self.SLA_THRESHOLD_MINUTES)
        return within / len(self._records)

    def average_csat(self) -> float:
        if not self._records:
            return 0.0
        return sum(r.csat_raw for r in self._records) / len(self._records)

    def first_call_resolution(self) -> float:
        if not self._records:
            return 0.0
        return sum(1 for r in self._records if not r.is_repeat) / len(self._records)

    def kpis_by_skill(self) -> Dict[str, Dict[str, float]]:
        skill_map: Dict[str, List[_CallRecord]] = {}
        for r in self._records:
            skill_map.setdefault(r.skill, []).append(r)
        return {
            skill: {
                "calls": len(records),
                "asa":   sum(r.wait_minutes   for r in records) / len(records),
                "aht":   sum(r.handle_minutes for r in records) / len(records),
                "csat":  sum(r.csat_raw       for r in records) / len(records),
                "sla":   sum(1 for r in records if r.wait_minutes <= self.SLA_THRESHOLD_MINUTES) / len(records),
            }
            for skill, records in skill_map.items()
        }

    def report(self) -> None:
        w   = 60
        bar = "=" * w
        print(f"\n{bar}")
        print(f"  CALL CENTRE SIMULATION -- KPI REPORT")
        print(f"{bar}")
        print(f"  {'Total calls handled':<35} {self.total_calls():>10}")
        print(f"  {'Total abandonments':<35} {self.total_abandonments():>10}")
        print(f"  {'Abandonment rate':<35} {self.abandonment_rate():>9.1%}")
        print(f"  {'SLA (<=1m)':<35} {self.sla_percentage():>9.1%}")
        print(f"  {'Avg speed of answer (ASA)':<35} {self.average_speed_of_answer():>8.2f}m")
        print(f"  {'Avg handle time (AHT)':<35} {self.average_handle_time():>8.2f}m")
        print(f"  {'Avg CSAT (1-5 scale)':<35} {self.average_csat():>8.2f}")
        print(f"  {'First-call resolution (FCR)':<35} {self.first_call_resolution():>9.1%}")
        print(f"\n  {'-' * (w - 4)}")
        print(f"  Breakdown by Skill")
        print(f"  {'-' * (w - 4)}")
        print(f"  {'Skill':<14} {'Calls':>6} {'ASA':>7} {'AHT':>7} {'CSAT':>6} {'SLA':>7}")
        print(f"  {'-' * (w - 4)}")
        for skill, m in sorted(self.kpis_by_skill().items()):
            print(
                f"  {skill:<14} {m['calls']:>6.0f} "
                f"{m['asa']:>6.2f}m {m['aht']:>6.2f}m "
                f"{m['csat']:>6.2f} {m['sla']:>6.1%}"
            )
        print(f"{bar}\n")


# ---------------------------------------------------------------------------
# SimulationEngine  — FIX 2: _pick_agent returns full Agent so bias is accessible
# ---------------------------------------------------------------------------

class SimulationEngine:
    _MAX_WAIT_PATIENCE: ClassVar[float] = 10.0

    def __init__(
        self,
        config:  SimulationConfig,
        weights: Optional[RouterScoreWeights] = None,
    ) -> None:
        self.config = config
        self.kpi    = KPIEngine()

        if config.random_seed is not None:
            random.seed(config.random_seed)

        self.env = simpy.Environment()
        self.agents = self._create_agent_pool()
        self._agent_map: Dict[str, Agent] = {a.agent_id: a for a in self.agents}
        self.skill_resources: Dict[str, simpy.PriorityResource] = {
            skill: simpy.PriorityResource(self.env, capacity=count)
            for skill, count in config.agents_per_skill.items()
        }
        self.router = Router(self.agents, config, weights)
        self._call_counter = 0

    def run(self) -> None:
        self.env.process(self._arrival_process())
        self._schedule_breaks()
        self.env.run(until=self.config.sim_duration_minutes)

    def _arrival_process(self):
        while True:
            inter_arrival = random.expovariate(self.config.arrival_rate_per_minute)
            yield self.env.timeout(inter_arrival)
            call = self._generate_call()
            self.env.process(self._handle_call(call))

    def _handle_call(self, call: Call):
        resource, routing_reason = self.router.select_resource(call, self.skill_resources)
        request = resource.request(priority=call.priority)
        result  = yield request | self.env.timeout(self._MAX_WAIT_PATIENCE)

        if request not in result:
            request.cancel()
            self.kpi.record_abandonment(call, self.env.now)
            return

        service_start = self.env.now
        agent         = self._pick_agent(call.skill)   # FIX: full Agent object
        agent_id      = agent.agent_id if agent else None
        if agent_id:
            self.router.notify_call_started(agent_id)

        service_time = max(
            0.5, random.gauss(self.config.mean_service_minutes, self.config.stdev_service_minutes)
        )
        acw_time = max(
            0.0, random.gauss(self.config.acw_mean_minutes, self.config.acw_stdev_minutes)
        )
        yield self.env.timeout(service_time + acw_time)

        resource.release(request)
        service_end    = self.env.now
        # FIX 3: pass agent bias into CSAT sampling so agents differ
        csat_raw       = self._sample_csat(call.customer_type, agent.csat_bias if agent else 0.0)
        handle_minutes = service_end - service_start

        if agent_id:
            self.router.notify_call_ended(
                agent_id, csat_raw, handle_minutes, call.is_repeat, skill=call.skill
            )

        self.kpi.record_call(_CallRecord(
            call_id       =call.call_id,
            skill         =call.skill,
            customer_type =call.customer_type,
            is_repeat     =call.is_repeat,
            arrival_time  =call.arrival_time,
            service_start =service_start,
            service_end   =service_end,
            csat_raw      =csat_raw,
            routing_reason=routing_reason,
        ))

    def _break_process(self, agent: Agent, start: float, duration: float):
        yield self.env.timeout(max(0.0, start - self.env.now))
        agent.on_break = True
        yield self.env.timeout(duration)
        agent.on_break = False

    def _create_agent_pool(self) -> List[Agent]:
        # FIX 4: assign csat_bias from N(0, 0.5) — large enough that ML
        # can learn which agents are genuinely better
        experience_distribution = ["junior", "junior", "mid", "mid", "senior"]
        pool:   List[Agent] = []
        skills = list(self.config.agents_per_skill.keys())
        for skill, count in self.config.agents_per_skill.items():
            other_skills = [s for s in skills if s != skill]
            for i in range(count):
                exp      = experience_distribution[i % len(experience_distribution)]
                secondary = (
                    [random.choice(other_skills)] if exp == "senior" and other_skills else []
                )
                csat_bias = random.gauss(0.0, 0.5)   # ← NEW: per-agent quality signal
                pool.append(Agent(
                    agent_id        =f"{skill[:3].upper()}-{i + 1:02d}",
                    primary_skill   =skill,
                    secondary_skills=secondary,
                    experience      =exp,
                    csat_bias       =csat_bias,
                ))
        return pool

    def _schedule_breaks(self) -> None:
        for agent in self.agents:
            for start, duration in self.config.break_schedule:
                self.env.process(self._break_process(agent, start, duration))

    def _generate_call(self) -> Call:
        self._call_counter += 1
        return Call(
            call_id      =f"CALL-{self._call_counter:06d}",
            skill        =self._weighted_choice(self.config.skill_mix),
            customer_type=self._weighted_choice(self.config.customer_tier_mix),
            is_repeat    =random.random() < self.config.repeat_call_probability,
            arrival_time =self.env.now,
        )

    def _pick_agent(self, skill: str) -> Optional[Agent]:
        """Return full Agent object (not just id) so bias flows through."""
        candidates = [a for a in self.agents if a.primary_skill == skill and not a.on_break]
        return random.choice(candidates) if candidates else None

    # Keep legacy method name for any subclasses that call it
    def _pick_agent_id(self, skill: str) -> Optional[str]:
        agent = self._pick_agent(skill)
        return agent.agent_id if agent else None

    @staticmethod
    def _weighted_choice(distribution: Dict[str, float]) -> str:
        keys, weights = list(distribution.keys()), list(distribution.values())
        total      = sum(weights)
        r          = random.uniform(0, total)
        cumulative = 0.0
        for key, w in zip(keys, weights):
            cumulative += w
            if r <= cumulative:
                return key
        return keys[-1]

    @staticmethod
    def _sample_csat(customer_type: str, agent_bias: float = 0.0) -> float:
        """FIX 5: agent_bias shifts baseline so agents produce genuinely different CSAT."""
        base = {"vip": 4.2, "premium": 3.9, "standard": 3.6}.get(customer_type, 3.6)
        return max(1.0, min(5.0, random.gauss(base + agent_bias, 0.8)))


# ===========================================================================
# HUMAN REALISM LAYER
# ===========================================================================

@dataclass
class BehaviorConfig:
    fatigue_rate:             float = 0.018
    fatigue_ceiling:          float = 0.85
    max_fatigue_penalty:      float = 0.15
    fatigue_csat_drag:        float = 0.10
    recovery_rate_per_min:    float = 0.012
    learning_rate:            float = 0.35
    max_learning_gain:        float = 0.15
    min_calls_to_plateau:     int   = 40
    break_start_delay_mean:   float = 2.0
    break_start_delay_stdev:  float = 3.5
    break_extension_prob:     float = 0.25
    break_extension_mean_min: float = 4.0
    early_return_prob:        float = 0.15
    early_return_frac:        float = 0.80


class FatigueModel:
    JITTER_SCALE:          float = 2.0
    RATE_VARIANCE:         float = 0.20
    RECOVERY_NONLINEARITY: float = 0.40

    _JITTER: tuple = (
        0.000, 0.006, 0.012, 0.018, 0.003, 0.009, 0.015,
        0.002, 0.008, 0.014, 0.005, 0.011, 0.017, 0.001,
    )
    _JITTER_MID: float = 0.009

    def __init__(self, cfg: BehaviorConfig, slot: int = 0) -> None:
        self._cfg     = cfg
        self._level   = 0.0
        jitter        = self._JITTER[slot % len(self._JITTER)]
        self._ceiling = min(1.0, cfg.fatigue_ceiling + jitter * self.JITTER_SCALE)
        rate_mult     = 1.0 + (jitter - self._JITTER_MID) * self.RATE_VARIANCE / self._JITTER_MID
        self._eff_rate = max(0.001, cfg.fatigue_rate * rate_mult)

    @property
    def level(self) -> float:
        return self._level

    def accumulate(self, handle_minutes: float) -> None:
        headroom    = max(0.0, self._ceiling - self._level)
        delta       = self._eff_rate * handle_minutes * headroom
        self._level = min(self._ceiling, self._level + delta)

    def recover(self, break_minutes: float) -> None:
        k           = self.RECOVERY_NONLINEARITY
        boost       = 1.0 + k * (1.0 - self._level)
        recovered   = self._cfg.recovery_rate_per_min * break_minutes * boost
        self._level = max(0.0, self._level - recovered)

    def handle_time_multiplier(self) -> float:
        return 1.0 + self._level * self._cfg.max_fatigue_penalty

    def csat_drag(self) -> float:
        return self._level * self._cfg.fatigue_csat_drag


class LearningCurveModel:
    def __init__(self, cfg: BehaviorConfig) -> None:
        self._cfg     = cfg
        self._n_calls = 0

    @property
    def calls_completed(self) -> int:
        return self._n_calls

    def record_call(self) -> None:
        self._n_calls += 1

    def csat_gain(self) -> float:
        n = self._n_calls
        if n == 0:
            return 0.0
        plateau = self._cfg.min_calls_to_plateau / 0.9
        return float(min(self._cfg.max_learning_gain,
                         self._cfg.max_learning_gain * (n / (n + plateau))))

    def handle_time_multiplier(self) -> float:
        gain_frac = self.csat_gain() / max(1e-6, self._cfg.max_learning_gain)
        return max(0.90, 1.0 - 0.10 * gain_frac)


class BreakVariabilityModel:
    def __init__(self, cfg: BehaviorConfig, rng_seed: Optional[int] = None) -> None:
        self._cfg    = cfg
        self._rng    = random.Random(rng_seed)
        self._events: List[Dict] = []

    def sample_break_timing(
        self, agent_id: str, scheduled_start: float, scheduled_duration: float
    ) -> Tuple[float, float]:
        cfg          = self._cfg
        delay        = max(0.0, self._rng.gauss(cfg.break_start_delay_mean, cfg.break_start_delay_stdev))
        actual_start = scheduled_start + delay
        effect       = "none"
        if self._rng.random() < cfg.early_return_prob:
            actual_duration = scheduled_duration * cfg.early_return_frac
            effect = "early_return"
        elif self._rng.random() < cfg.break_extension_prob:
            extra           = self._rng.expovariate(1.0 / cfg.break_extension_mean_min)
            actual_duration = scheduled_duration + extra
            effect          = f"extended+{extra:.1f}m"
        else:
            actual_duration = scheduled_duration
        self._events.append({
            "agent_id": agent_id, "sched_start": scheduled_start,
            "sched_dur": scheduled_duration, "delay": delay,
            "actual_dur": actual_duration, "effect": effect,
        })
        return actual_start, actual_duration

    def print_break_log(self) -> None:
        if not self._events:
            print("  [BreakVariabilityModel] No break events recorded.")
            return
        print("\n-- Break Variability Log -------------------------------------------")
        print(f"  {'Agent':<20}  {'Sched Start':>12}  {'Delay':>7}  {'Sched Dur':>10}  {'Actual Dur':>10}  Effect")
        print("  " + "-" * 76)
        for ev in self._events:
            print(
                f"  {ev['agent_id']:<20}  {ev['sched_start']:>12.1f}  "
                f"{ev['delay']:>6.1f}m  {ev['sched_dur']:>9.1f}m  "
                f"{ev['actual_dur']:>9.1f}m  {ev['effect']}"
            )


class HumanAgentState:
    def __init__(self, agent_id: str, cfg: BehaviorConfig, slot: int = 0) -> None:
        self.agent_id  = agent_id
        self.cfg       = cfg
        self.fatigue   = FatigueModel(cfg, slot=slot)
        self.learning  = LearningCurveModel(cfg)
        self.total_handle_minutes: float = 0.0
        self.peak_fatigue_level:   float = 0.0
        self.calls_handled:        int   = 0

    def adjust_service_time(self, base_minutes: float) -> float:
        return max(0.5, base_minutes
                   * self.fatigue.handle_time_multiplier()
                   * self.learning.handle_time_multiplier())

    def adjust_csat(self, base_csat: float) -> float:
        norm     = (base_csat - 1.0) / 4.0
        adjusted = norm - self.fatigue.csat_drag() + self.learning.csat_gain()
        return 1.0 + max(0.0, min(1.0, adjusted)) * 4.0

    def on_call_end(self, handle_minutes: float) -> None:
        self.fatigue.accumulate(handle_minutes)
        self.learning.record_call()
        self.total_handle_minutes += handle_minutes
        self.calls_handled        += 1
        self.peak_fatigue_level    = max(self.peak_fatigue_level, self.fatigue.level)

    def on_break_end(self, break_duration: float) -> None:
        self.fatigue.recover(break_duration)


class HumanRealisticsEngine:
    def __init__(self, agents: List[Agent], cfg: SimulationConfig, beh: BehaviorConfig) -> None:
        self._beh = beh
        self._states: Dict[str, HumanAgentState] = {
            a.agent_id: HumanAgentState(a.agent_id, beh, slot=i)
            for i, a in enumerate(agents)
        }
        self._break_model = BreakVariabilityModel(beh)

    def adjust_service_times(
        self, agent_id: str, service_time: float, acw_time: float
    ) -> Tuple[float, float]:
        state = self._states.get(agent_id)
        if state is None:
            return service_time, acw_time
        return (state.adjust_service_time(service_time),
                state.adjust_service_time(acw_time))

    def adjust_csat(self, agent_id: str, csat_raw: float) -> float:
        state = self._states.get(agent_id)
        return state.adjust_csat(csat_raw) if state else csat_raw

    def on_call_end(self, agent_id: str, handle_minutes: float) -> None:
        state = self._states.get(agent_id)
        if state:
            state.on_call_end(handle_minutes)

    def on_break_end(self, agent_id: str, actual_duration: float) -> None:
        state = self._states.get(agent_id)
        if state:
            state.on_break_end(actual_duration)

    def sample_break_timing(
        self, agent_id: str, scheduled_start: float, scheduled_duration: float
    ) -> Tuple[float, float]:
        return self._break_model.sample_break_timing(
            agent_id, scheduled_start, scheduled_duration
        )

    def fatigue_summary(self) -> Dict[str, float]:
        return {aid: st.fatigue.level for aid, st in self._states.items()}

    def learning_summary(self) -> Dict[str, float]:
        return {aid: st.learning.csat_gain() for aid, st in self._states.items()}

    def print_report(self) -> None:
        w   = 72
        bar = "=" * w
        print(f"\n{bar}")
        print(f"  HUMAN REALISM LAYER -- AGENT STATE REPORT")
        print(f"{bar}")
        print(
            f"  {'Agent':<20}  {'Calls':>6}  {'Fatigue':>8}  "
            f"{'PeakFat':>8}  {'LrnGain':>8}  {'TotalHdl':>9}"
        )
        print(f"  {'-' * (w - 4)}")
        for state in sorted(self._states.values(), key=lambda s: s.agent_id):
            print(
                f"  {state.agent_id:<20}  {state.calls_handled:>6}  "
                f"{state.fatigue.level:>8.3f}  {state.peak_fatigue_level:>8.3f}  "
                f"{state.learning.csat_gain():>8.4f}  {state.total_handle_minutes:>8.1f}m"
            )
        print(f"{bar}")
        self._break_model.print_break_log()


class RealisticsAwareEngine(SimulationEngine):
    """SimulationEngine extended with the human realism layer."""

    def __init__(
        self,
        config:   SimulationConfig,
        behavior: Optional[BehaviorConfig] = None,
        weights:  Optional[RouterScoreWeights] = None,
    ) -> None:
        super().__init__(config, weights)
        self.realism = HumanRealisticsEngine(
            self.agents, config, behavior or BehaviorConfig()
        )

    def _handle_call(self, call: Call):
        resource, routing_reason = self.router.select_resource(call, self.skill_resources)
        request = resource.request(priority=call.priority)
        result  = yield request | self.env.timeout(self._MAX_WAIT_PATIENCE)

        if request not in result:
            request.cancel()
            self.kpi.record_abandonment(call, self.env.now)
            return

        service_start = self.env.now
        agent         = self._pick_agent(call.skill)
        agent_id      = agent.agent_id if agent else None
        if agent_id:
            self.router.notify_call_started(agent_id)

        base_svc = max(
            0.5, random.gauss(self.config.mean_service_minutes, self.config.stdev_service_minutes)
        )
        base_acw = max(
            0.0, random.gauss(self.config.acw_mean_minutes, self.config.acw_stdev_minutes)
        )

        if agent_id:
            svc_time, acw_time = self.realism.adjust_service_times(agent_id, base_svc, base_acw)
        else:
            svc_time, acw_time = base_svc, base_acw

        yield self.env.timeout(svc_time + acw_time)

        resource.release(request)
        service_end    = self.env.now
        # Pass agent bias through realism path too
        base_csat      = self._sample_csat(call.customer_type, agent.csat_bias if agent else 0.0)
        csat_raw       = self.realism.adjust_csat(agent_id, base_csat) if agent_id else base_csat
        handle_minutes = service_end - service_start

        if agent_id:
            self.realism.on_call_end(agent_id, handle_minutes)
            self.router.notify_call_ended(
                agent_id, csat_raw, handle_minutes, call.is_repeat, skill=call.skill
            )

        self.kpi.record_call(_CallRecord(
            call_id       =call.call_id,
            skill         =call.skill,
            customer_type =call.customer_type,
            is_repeat     =call.is_repeat,
            arrival_time  =call.arrival_time,
            service_start =service_start,
            service_end   =service_end,
            csat_raw      =csat_raw,
            routing_reason=routing_reason,
        ))

    def _break_process(self, agent: Agent, start: float, duration: float):
        actual_start, actual_duration = self.realism.sample_break_timing(
            agent.agent_id, start, duration
        )
        yield self.env.timeout(max(0.0, actual_start - self.env.now))
        agent.on_break = True
        yield self.env.timeout(actual_duration)
        agent.on_break = False
        self.realism.on_break_end(agent.agent_id, actual_duration)