Code Flow: python main.py — Step by Step
This is a call centre staffing optimization pipeline with 5 stages. Here's exactly what happens:

🔧 Initialization (before main())
When Python loads the file, these module-level objects are created:
sim_cfg   → SimulationConfig(60 calls/hr, 720-min horizon, seed=42)
cost_cfg  → CostConfig(wages: billing=£18, technical=£22, general=£16)
beh_cfg   → BehaviorConfig(fatigue/learning/break variability params)

main() is called
Prints a summary banner, then runs 6 stages sequentially.

📊 STAGE 0 — Erlang-C Analytical Preview
Pure math, no simulation.

Takes arrival rate (60/hr = 1.0/min), mean service time (5.0 + 1.5 = 6.5 min)
For each skill (billing 40%, technical 35%, general 25%), calculates the peak-band arrival rate

Peak band = minutes 240–480 (all 3 shifts overlap here)
Peak fraction = 240/720 = 0.333


Runs ErlangC.min_agents_for_sla() — loops c = 1, 2, 3... until Erlang-C formula says SLA ≥ 90%
Prints a per-band scan showing minimum agents needed at every time window

Output: A table showing how many agents each skill needs analytically.

🧮 STAGE 1 — Single CP-SAT Solve
Optimization only, no simulation.

Creates OptimizationConfig (SLA=90%, buffer=1.10x, max_occupancy=85%)
StaffingOptimizer.solve() is called:

Calls _build_band_erlang_mins() → computes minimum agents needed per band per skill (5 bands × 3 skills = 15 constraints)
Builds a CP-SAT model (Google OR-Tools):

Variables: n_M_billing, n_M_technical, ... (3 shifts × 3 skills = 9 integer variables)
Constraints: coverage in each band ≥ Erlang-C minimum
Objective: minimize total wage cost


Calls cp_model.CpSolver().Solve() — typically finds optimal in milliseconds


Extracts shift_plan → e.g. {M: {billing:3, technical:3, general:2}, D: {...}, E: {...}}
Prints the shift plan and Erlang-C SLA per skill

Output: Cheapest analytically-valid staffing plan.

🔁 STAGE 2 — Optimize → Simulate Loop
The core loop: solve analytically, validate with simulation, repeat.
Iterates up to 8 times:
Each iteration:

StaffingOptimizer.solve() — same CP-SAT solve as Stage 1, but with an increasing analytical_target (starts at 90%, bumps up if simulation misses)
SimulationEvaluator.evaluate() — runs a full discrete-event simulation:

Creates SimulationConfig with optimized agent counts
Builds CostAwareEngine (SimPy-based)
engine.run() kicks off:

Arrival process: generates calls via exponential inter-arrivals (Poisson process)
Each call gets: random skill, customer tier, is_repeat flag
_handle_call() for each call:

Router.select_resource() picks which skill queue to join (VIP fast-path, overflow if queue deep, else best-scored agent)
Call waits in PriorityResource queue (VIPs jump ahead)
If waits > 10 min → abandons (recorded in KPI)
Otherwise: agent assigned via Router.pick_agent() (scores agents on CSAT, handle time, workload, etc.)
Service time drawn from Normal(5.0, 2.0), ACW from Normal(1.5, 0.5)
CSAT score sampled (VIP base=4.2, premium=3.9, standard=3.6 + agent's personal bias)


Break process: at minutes 240, 360, 480 — holds one server slot at low priority so breaks never starve calls


After run: settles staffing, overtime, idle, burnout costs


Gap check: sim_SLA - 90%

If gap ≥ -1.5% → converged, stop
Else: increase analytical target by min(10%, gap × 1.5) adaptively



After loop:

Runs Pareto sweep (cost-only vs balanced vs SLA-heavy weights)
Calls explain_plan() and print_report() with full breakdown

Output: Best shift plan that actually achieves ≥90% SLA in simulation.

🤖 STAGE 3 — ML Feedback Loop
Train ML models on simulation data, improve routing over 5 epochs.

MLModelRegistry.warm_start() — pre-trains models on synthetic data:

CSATPredictionModel → XGBoost (or SGD fallback) predicting if CSAT ≥ 4
AbandonmentRiskModel → LightGBM + Weibull survival model
FCRPredictionModel → XGBoost for first-call resolution
IntradayArrivalForecaster → LightGBM quantile regression


5 epochs, each epoch:

Reseeds RNG (seed = 42 + epoch × 17) for reproducibility
Resets FatigueTracker so fatigue doesn't carry across epochs
Runs MLSimulationEngine — same as SimPy engine but uses MLRouter instead of base Router

MLRouter._composite_score() blends:

40% base heuristic score (CSAT history, handle time, workload)
60% ML signals:

CSAT prediction (30%)
FCR prediction (20%)
Abandonment risk (20%)
Contextual affinity EMA (20%)
Fatigue penalty (10%)






Collects training data from call records → CSAT dataset, abandonment dataset, FCR dataset
Retrains all models on accumulated data
Checks convergence (if SLA/CSAT stops improving for 2 patience epochs → early stop)



Output: Trained ML models + agent affinity table (which agents handle which skill/tier well).

🏭 STAGE 4 — Full-Stack Evaluation
Runs two engines side-by-side with the optimized staffing.
Stage 4a: CostAwareRealisticsEngine
Adds the Human Realism Layer on top of everything:

FatigueModel: each call accumulates fatigue → handle time multiplier increases (up to +15%), CSAT drag increases
LearningCurveModel: first 40 calls → agents get faster and better (up to -10% handle time, +15% CSAT)
BreakVariabilityModel: breaks have random delays (mean +2min), 25% chance of extension, 15% chance of early return

Every call goes through:
base_service_time
  × fatigue_multiplier  (1.0 → 1.15 as shift progresses)
  × learning_multiplier (1.0 → 0.90 as agent gains experience)
→ actual_service_time

base_CSAT
  - fatigue_drag  (up to -0.10)
  + learning_gain (up to +0.15 normalized)
→ actual_CSAT
After run: prints KPI report, cost breakdown, agent performance stats, human realism state.
Stage 4b: MLSimulationEngine
Same staffing but using trained ML router. Prints side-by-side comparison table.

📈 STAGE 5 — Scenario Comparison
Runs two CostAwareRealisticsEngine instances:
Default configOptimized configAgents11 (4+4+3)~15 (from Stage 2)Horizon720 min720 min
Computes and prints:

SLA delta, abandonment delta, CSAT delta, ASA delta
Total cost delta — if optimized plan costs more in staffing but saves more in SLA violations + abandonment penalties, it shows net saving


🏁 Pipeline Complete
Prints final summary:

Best peak-band plan (e.g. {billing: 5, technical: 5, general: 4})
Full shift plan (M/D/E breakdown)
Total runtime in seconds


Key Data Flow Summary
Erlang-C math
    ↓
CP-SAT solver → shift_plan (agents per shift per skill)
    ↓
SimPy simulation → real SLA
    ↓ (feedback: if SLA < target, raise analytical target)
Loop until converged
    ↓
ML training on call records (XGBoost/LightGBM)
    ↓
ML-routed simulation (smarter agent selection)
    ↓
Realism-aware simulation (fatigue + learning + breaks)
    ↓
Cost comparison (default vs optimized staffing)
The whole pipeline is essentially: "find cheapest staffing that actually meets SLA when human behavior is modeled realistically."