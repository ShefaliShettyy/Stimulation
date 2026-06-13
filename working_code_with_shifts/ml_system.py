from __future__ import annotations

import copy
import math
import random
import time
import warnings
from dataclasses import dataclass, field
from typing import (
    Callable, ClassVar, Dict, List, Literal,
    Optional, Tuple, Any,
)

import numpy as np
from scipy.stats import weibull_min

from core_simulation import (
    Agent, Call, Router, RouterScoreWeights,
    SimulationConfig, SimulationEngine, _CallRecord,
)

try:
    from cost_system import AgentUtilizationCollector, AgentUtilizationStats
    _COST_AVAILABLE = True
except ImportError:
    _COST_AVAILABLE = False

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False

try:
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.preprocessing import StandardScaler
    _SKL_AVAILABLE = True
except ImportError:
    _SKL_AVAILABLE = False

# Narrow warning suppression: only silence scikit-learn convergence warnings
# (which are expected for small online fits) rather than every UserWarning.
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:  # pragma: no cover - sklearn not installed
    pass
# Models are fit and queried with plain NumPy arrays; silence only the specific
# (harmless) sklearn/LightGBM message about missing feature names rather than
# every UserWarning.
warnings.filterwarnings("ignore", message="X does not have valid feature names")


# ---------------------------------------------------------------------------
# REPRO helper
# ---------------------------------------------------------------------------

def _reset_numpy_seed(seed: int) -> None:
    np.random.seed(int(seed) & 0xFFFFFFFF)


# ===========================================================================
# HELPERS
# ===========================================================================

def _skill_to_int(skill: str) -> int:
    return {"billing": 0, "technical": 1, "general": 2}.get(skill, 0)

def _tier_to_int(tier: str) -> int:
    return {"standard": 0, "premium": 1, "vip": 2}.get(tier, 0)

def _exp_to_int(exp: str) -> int:
    return {"junior": 0, "mid": 1, "senior": 2}.get(exp, 1)

def _complexity_bucket(handle_minutes: float) -> str:
    if handle_minutes < 4.0:
        return "simple"
    if handle_minutes < 9.0:
        return "medium"
    return "complex"

def _complexity_to_int(handle_minutes: float) -> int:
    return {"simple": 0, "medium": 1, "complex": 2}[_complexity_bucket(handle_minutes)]


# ===========================================================================
# PART 0 — DRIFT DETECTION
# ===========================================================================

class PageHinkleyDrift:
    def __init__(self, delta: float = 0.005, lambda_: float = 50.0) -> None:
        self._delta   = delta
        self._lambda  = lambda_
        self._sum     = 0.0
        self._min_sum = 0.0
        self._n       = 0
        self._mean    = 0.0
        self.drift_detected = False

    def update(self, x: float) -> bool:
        self._n   += 1
        self._mean += (x - self._mean) / self._n
        self._sum  += x - self._mean - self._delta
        self._min_sum = min(self._min_sum, self._sum)
        self.drift_detected = (self._sum - self._min_sum) > self._lambda
        return self.drift_detected

    def reset(self) -> None:
        self._sum = 0.0
        self._min_sum = 0.0
        self._n = 0
        self._mean = 0.0
        self.drift_detected = False


# ===========================================================================
# PART 1 — ML MODELS
# ===========================================================================

class CSATPredictionModel:
    N_FEATURES: ClassVar[int] = 8

    def __init__(self) -> None:
        self._xgb_model: Optional[Any]  = None
        self._sgd_model: Optional[Any]  = None
        self._scaler:    Optional[Any]  = None
        self._fitted     = False
        self._n_calls    = 0
        self._drift      = PageHinkleyDrift(delta=0.005, lambda_=40.0)
        self._warm_w     = np.array([0.10, 0.05, -0.30, 2.00, -0.08, 0.40, 0.10, -0.05])
        self._warm_b     = -0.5

    def warm_start(self) -> "CSATPredictionModel":
        rng   = np.random.default_rng(42)
        X_neg = np.column_stack([
            rng.integers(0, 2, 60), rng.integers(0, 3, 60), np.ones(60),
            rng.uniform(0.0, 0.3, 60), rng.uniform(8.0, 15.0, 60),
            np.zeros(60), rng.integers(0, 2, 60), rng.integers(1, 4, 60),
        ])
        X_pos = np.column_stack([
            rng.integers(1, 3, 60), rng.integers(0, 3, 60), np.zeros(60),
            rng.uniform(0.7, 1.0, 60), rng.uniform(3.0, 7.0, 60),
            np.ones(60), rng.integers(1, 3, 60), rng.integers(0, 2, 60),
        ])
        X = np.vstack([X_neg, X_pos])
        y = np.array([0] * 60 + [1] * 60, dtype=int)
        _reset_numpy_seed(42)
        self.fit(X, y)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) < 10 or len(np.unique(y)) < 2:
            return
        if _XGB_AVAILABLE:
            self._xgb_model = xgb.XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.08,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                verbosity=0, eval_metric="logloss",
            )
            self._xgb_model.fit(X, y)
        elif _SKL_AVAILABLE:
            self._scaler    = StandardScaler().fit(X)
            self._sgd_model = SGDClassifier(loss="log_loss", random_state=42, max_iter=200)
            self._sgd_model.fit(self._scaler.transform(X), y)
        self._fitted  = True
        self._n_calls += len(X)

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if not _SKL_AVAILABLE or len(X) < 5 or len(np.unique(y)) < 2:
            return
        if self._sgd_model is None:
            if self._scaler is None:
                self._scaler = StandardScaler()
                Xt = self._scaler.fit_transform(X)
            else:
                Xt = self._scaler.transform(X)
            self._sgd_model = SGDClassifier(loss="log_loss", random_state=42)
            self._sgd_model.partial_fit(Xt, y, classes=[0, 1])
        else:
            Xt = self._scaler.transform(X)
            self._sgd_model.partial_fit(Xt, y)

    def predict_proba_positive(self, features: np.ndarray) -> float:
        x = np.asarray(features, dtype=float).reshape(1, -1)
        if self._xgb_model is not None and self._fitted:
            try:
                return float(self._xgb_model.predict_proba(x)[0, 1])
            except Exception:
                pass
        if self._sgd_model is not None:
            try:
                Xt = self._scaler.transform(x)
                return float(self._sgd_model.predict_proba(Xt)[0, 1])
            except Exception:
                pass
        score = float(np.dot(x.ravel(), self._warm_w)) + self._warm_b
        return float(1.0 / (1.0 + np.exp(-score)))

    def update_drift(self, observed_csat: float) -> bool:
        return self._drift.update(observed_csat / 5.0)

    def print_report(self) -> None:
        backend = "XGBoost" if self._xgb_model else ("SGD" if self._sgd_model else "heuristic")
        fitted  = "fitted" if self._fitted else "not fitted"
        print(f"  CSATPredictionModel      [{backend}, {fitted}, n={self._n_calls}]")
        if self._xgb_model is not None and hasattr(self._xgb_model, "feature_importances_"):
            imp = np.round(self._xgb_model.feature_importances_, 3)
            print(f"    Feature importances: {imp}")


# ---------------------------------------------------------------------------
# Abandonment Risk
# ---------------------------------------------------------------------------

class SurvivalAbandonmentModel:
    _DEFAULTS: ClassVar[Dict[str, Tuple[float, float]]] = {
        "standard": (1.2, 3.0),
        "premium":  (1.4, 5.0),
        "vip":      (1.6, 8.0),
    }

    def __init__(self) -> None:
        self._params: Dict[str, Tuple[float, float]] = dict(self._DEFAULTS)
        self._fitted = False

    def fit(self, wait_times: List[float], abandoned_flags: List[int], tiers: List[str]) -> None:
        for tier in set(tiers):
            mask      = [t == tier for t in tiers]
            wts       = [w for w, m in zip(wait_times, mask) if m]
            abns      = [a for a, m in zip(abandoned_flags, mask) if m]
            abn_waits = [w for w, a in zip(wts, abns) if a == 1]
            if len(abn_waits) < 5:
                continue
            mu    = float(np.mean(abn_waits))
            if mu <= 0.0:
                # Degenerate all-zero waits: keep this tier's default params to
                # avoid a divide-by-zero / zero-scale Weibull.
                continue
            std   = float(np.std(abn_waits)) + 1e-9
            cv    = std / mu
            k     = float(np.clip((1.0 / (cv + 0.01)) ** 1.086, 0.5, 5.0))
            scale = mu / math.gamma(1.0 + 1.0 / k)
            self._params[tier] = (k, scale)
        self._fitted = True

    def abandon_probability(self, wait_minutes: float, tier: str = "standard") -> float:
        k, scale = self._params.get(tier, self._params["standard"])
        return float(weibull_min.cdf(max(0.0, wait_minutes), c=k, scale=scale))

    def expected_patience(self, tier: str = "standard") -> float:
        k, scale = self._params.get(tier, self._params["standard"])
        return scale * math.gamma(1.0 + 1.0 / k)

    def print_report(self) -> None:
        print(f"  SurvivalAbandonmentModel [fitted={self._fitted}]")
        for tier, (k, sc) in self._params.items():
            ep = sc * math.gamma(1 + 1/k)
            print(f"    {tier:<10} k={k:.2f} scale={sc:.2f} E[patience]={ep:.1f}m")


class AbandonmentRiskModel:
    N_FEATURES: ClassVar[int] = 6

    def __init__(self) -> None:
        self._survival   = SurvivalAbandonmentModel()
        self._lgbm_model: Optional[Any] = None
        self._sgd_model:  Optional[Any] = None
        self._scaler:     Optional[Any] = None
        self._fitted      = False
        self._n_calls     = 0
        self._drift       = PageHinkleyDrift(delta=0.01, lambda_=30.0)
        self._warm_w = np.array([0.25, -0.15, 0.05, 0.10, 0.02, -0.001])
        self._warm_b = -1.5

    def warm_start(self) -> "AbandonmentRiskModel":
        rng   = np.random.default_rng(42)
        X_neg = np.column_stack([
            rng.uniform(0.0, 0.5, 60), rng.integers(1, 3, 60),
            rng.integers(0, 3, 60), np.zeros(60),
            rng.integers(0, 3, 60), rng.uniform(0, 480, 60),
        ])
        X_pos = np.column_stack([
            rng.uniform(5.0, 12.0, 60), np.zeros(60),
            rng.integers(0, 3, 60), rng.integers(0, 2, 60),
            rng.integers(3, 8, 60), rng.uniform(0, 480, 60),
        ])
        X = np.vstack([X_neg, X_pos])
        y = np.array([0] * 60 + [1] * 60, dtype=int)
        _reset_numpy_seed(42)
        self.fit(X, y)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) < 10 or len(np.unique(y)) < 2:
            return
        if _LGB_AVAILABLE:
            self._lgbm_model = lgb.LGBMClassifier(
                n_estimators=150, num_leaves=15, learning_rate=0.08,
                is_unbalance=True, random_state=42, verbose=-1,
            )
            self._lgbm_model.fit(X, y)
        elif _SKL_AVAILABLE:
            self._scaler    = StandardScaler().fit(X)
            self._sgd_model = SGDClassifier(loss="log_loss", random_state=42)
            self._sgd_model.fit(self._scaler.transform(X), y)
        self._fitted  = True
        self._n_calls += len(X)

    def fit_survival(self, wait_times: List[float], abandoned_flags: List[int], tiers: List[str]) -> None:
        self._survival.fit(wait_times, abandoned_flags, tiers)

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if not _SKL_AVAILABLE or len(X) < 5 or len(np.unique(y)) < 2:
            return
        if self._sgd_model is None:
            self._scaler    = StandardScaler().fit(X) if self._scaler is None else self._scaler
            self._sgd_model = SGDClassifier(loss="log_loss", random_state=42)
            self._sgd_model.partial_fit(self._scaler.transform(X), y, classes=[0, 1])
        else:
            self._sgd_model.partial_fit(self._scaler.transform(X), y)

    def predict_abandon_prob(self, features: np.ndarray, tier: str = "standard", wait_minutes: float = 0.0) -> float:
        survival_p = self._survival.abandon_probability(wait_minutes, tier)
        x = np.asarray(features, dtype=float).reshape(1, -1)
        if self._lgbm_model is not None and self._fitted:
            try:
                lgbm_p = float(self._lgbm_model.predict_proba(x)[0, 1])
                return 0.6 * survival_p + 0.4 * lgbm_p
            except Exception:
                pass
        if self._sgd_model is not None:
            try:
                Xt    = self._scaler.transform(x)
                sgd_p = float(self._sgd_model.predict_proba(Xt)[0, 1])
                return 0.6 * survival_p + 0.4 * sgd_p
            except Exception:
                pass
        score  = float(np.dot(x.ravel(), self._warm_w)) + self._warm_b
        lgbm_p = float(1.0 / (1.0 + np.exp(-score)))
        return 0.6 * survival_p + 0.4 * lgbm_p

    def update_drift(self, observed_abandon_rate: float) -> bool:
        return self._drift.update(observed_abandon_rate)

    def print_report(self) -> None:
        backend = "LightGBM" if self._lgbm_model else ("SGD" if self._sgd_model else "heuristic")
        print(f"  AbandonmentRiskModel     [{backend}+Weibull, fitted={self._fitted}, n={self._n_calls}]")
        self._survival.print_report()


# ---------------------------------------------------------------------------
# FCR Prediction
# ---------------------------------------------------------------------------

class FCRPredictionModel:
    N_FEATURES: ClassVar[int] = 7

    def __init__(self) -> None:
        self._model:  Optional[Any] = None
        self._fitted  = False
        self._n_calls = 0
        self._warm_w  = np.array([0.05, 0.15, -0.30, 1.50, -0.10, 0.20, -0.25])
        self._warm_b  = -0.2

    def warm_start(self) -> "FCRPredictionModel":
        rng   = np.random.default_rng(42)
        X_neg = np.column_stack([
            rng.integers(0, 3, 60), rng.integers(0, 3, 60), np.ones(60),
            rng.uniform(0.0, 0.4, 60), rng.uniform(9.0, 15.0, 60),
            rng.integers(0, 2, 60), np.full(60, 2),
        ])
        X_pos = np.column_stack([
            rng.integers(0, 3, 60), rng.integers(0, 3, 60), np.zeros(60),
            rng.uniform(0.7, 1.0, 60), rng.uniform(4.0, 8.0, 60),
            rng.integers(1, 3, 60), rng.integers(0, 2, 60),
        ])
        X = np.vstack([X_neg, X_pos])
        y = np.array([0] * 60 + [1] * 60, dtype=int)
        _reset_numpy_seed(42)
        self.fit(X, y)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) < 10 or len(np.unique(y)) < 2:
            return
        if _XGB_AVAILABLE:
            self._model = xgb.XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=42, verbosity=0, eval_metric="logloss",
            )
            self._model.fit(X, y)
        elif _SKL_AVAILABLE:
            self._model = LogisticRegression(max_iter=200, random_state=42)
            self._model.fit(X, y)
        self._fitted  = True
        self._n_calls += len(X)

    def predict_fcr_prob(self, features: np.ndarray) -> float:
        x = np.asarray(features, dtype=float).reshape(1, -1)
        if self._model is not None and self._fitted:
            try:
                return float(self._model.predict_proba(x)[0, 1])
            except Exception:
                pass
        score = float(np.dot(x.ravel(), self._warm_w)) + self._warm_b
        return float(1.0 / (1.0 + np.exp(-score)))

    def print_report(self) -> None:
        backend = "XGBoost" if (_XGB_AVAILABLE and self._model) else "heuristic"
        print(f"  FCRPredictionModel       [{backend}, fitted={self._fitted}, n={self._n_calls}]")


# ---------------------------------------------------------------------------
# Intraday Arrival Forecaster
# ---------------------------------------------------------------------------

class IntradayArrivalForecaster:
    N_INTERVALS:    ClassVar[int]   = 32
    N_FEATURES:     ClassVar[int]   = 10
    DEFAULT_HORIZON: ClassVar[float] = 480.0

    def __init__(self) -> None:
        self._models:    Dict[float, Any] = {}
        self._base_rate: float            = 1.0
        self._fitted     = False
        self._history:   List[np.ndarray] = []

    def _interval_minutes(self, horizon: float) -> float:
        return horizon / self.N_INTERVALS

    def warm_start_synthetic(self) -> "IntradayArrivalForecaster":
        rng = np.random.default_rng(0)
        t   = np.linspace(0, self.N_INTERVALS - 1, self.N_INTERVALS)
        history = [
            np.clip(
                self._base_rate * (0.8 + 0.4 * np.sin(np.pi * t / self.N_INTERVALS))
                + rng.normal(0, 0.08, self.N_INTERVALS),
                0.1, 10.0,
            )
            for _ in range(14)
        ]
        self.fit(history)
        return self

    def _build_row(self, interval_idx: int, dow: int, lag4: List[float]) -> np.ndarray:
        angle = 2.0 * math.pi * interval_idx / self.N_INTERVALS
        return np.array([
            float(interval_idx), math.sin(angle), math.cos(angle),
            float(dow), float(dow == 0), float(dow == 4),
            *lag4,
        ], dtype=float)

    def fit(self, history: List[np.ndarray]) -> None:
        if not _LGB_AVAILABLE or len(history) < 3:
            self._fitted = False
            return
        self._history = list(history)
        X, y = [], []
        for d, rates in enumerate(history):
            dow = d % 7
            for t in range(4, self.N_INTERVALS):
                lag4 = list(rates[t - 4: t])
                X.append(self._build_row(t, dow, lag4))
                y.append(float(rates[t]))
        X_arr = np.array(X); y_arr = np.array(y)
        for q in (0.10, 0.50, 0.90):
            m = lgb.LGBMRegressor(
                objective="quantile", alpha=q, n_estimators=100,
                num_leaves=15, learning_rate=0.05, random_state=42, verbose=-1,
            )
            m.fit(X_arr, y_arr)
            self._models[q] = m
        self._fitted = True

    def predict_interval(self, interval_idx: int, dow: int = 0, lag4: Optional[List[float]] = None) -> Tuple[float, float, float]:
        lag4 = lag4 or [self._base_rate] * 4
        row  = self._build_row(interval_idx, dow, lag4).reshape(1, -1)
        if self._fitted:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p10 = float(self._models[0.10].predict(row)[0])
                    p50 = float(self._models[0.50].predict(row)[0])
                    p90 = float(self._models[0.90].predict(row)[0])
                p10 = max(0.0, p10); p50 = max(p10, p50); p90 = max(p50, p90)
                return p10, p50, p90
            except Exception:
                pass
        base = self._base_rate * (0.8 + 0.4 * math.sin(math.pi * interval_idx / self.N_INTERVALS))
        return max(0.0, base * 0.8), base, base * 1.25

    def predict_shift(self, dow: int = 0, use_quantile: float = 0.90,
                      horizon: float = DEFAULT_HORIZON) -> List[Dict[str, float]]:
        interval_min = self._interval_minutes(horizon)
        result: List[Dict[str, float]] = []
        lag4 = [self._base_rate] * 4
        for t in range(self.N_INTERVALS):
            p10, p50, p90 = self.predict_interval(t, dow, lag4)
            result.append({
                "interval": t, "minute_start": t * interval_min,
                "p10": p10, "p50": p50, "p90": p90,
                "recommended": (p10 if use_quantile <= 0.10 else (p90 if use_quantile >= 0.90 else p50)),
            })
            lag4 = lag4[1:] + [p50]
        return result

    def append_shift(self, rates: np.ndarray) -> None:
        self._history.append(rates)
        if len(self._history) > 90:
            self._history = self._history[-90:]

    def predict(self, sim_time: float, horizon: float = DEFAULT_HORIZON) -> float:
        interval_idx = min(self.N_INTERVALS - 1, int(sim_time // self._interval_minutes(horizon)))
        _, p50, _ = self.predict_interval(interval_idx)
        return max(0.1, p50)

    def make_rate_fn(self, dow: int = 0, horizon: float = DEFAULT_HORIZON) -> Callable[[float], float]:
        forecast = self.predict_shift(dow=dow, use_quantile=0.50, horizon=horizon)
        interval_min = self._interval_minutes(horizon)

        def rate_fn(sim_time: float) -> float:
            idx = min(self.N_INTERVALS - 1, int(sim_time // interval_min))
            return max(1e-6, forecast[idx]["recommended"])

        return rate_fn

    def print_report(self) -> None:
        backend = "LightGBM quantile" if self._fitted else "heuristic"
        print(
            f"  IntradayArrivalForecaster [{backend}, base={self._base_rate:.3f}/min, "
            f"history={len(self._history)} shifts]"
        )


class ArrivalForecaster(IntradayArrivalForecaster):
    """Backward-compatible alias."""
    pass


# ---------------------------------------------------------------------------
# Contextual Agent Affinity
# ---------------------------------------------------------------------------

class ContextualAffinityModel:
    EMA_ALPHA: ClassVar[float] = 0.20

    def __init__(self) -> None:
        self._ctx_scores:   Dict[Tuple[str, str, str, str], float] = {}
        self._skill_scores: Dict[Tuple[str, str], float]           = {}

    def update(self, agent_id: str, skill: str, csat_raw: float, handle_minutes: float,
               is_repeat: bool, tier: str = "standard", queue_depth: int = 0) -> None:
        csat_norm    = (csat_raw - 1.0) / 4.0
        speed_norm   = max(0.0, min(1.0, 1.0 - handle_minutes / 15.0))
        res_signal   = 0.0 if is_repeat else 1.0
        pressure_pen = min(0.10, queue_depth * 0.01)
        signal = max(0.0, min(1.0,
            0.45 * csat_norm + 0.25 * speed_norm + 0.20 * res_signal - pressure_pen
        ))
        complexity = _complexity_bucket(handle_minutes)
        ctx_key = (agent_id, skill, tier, complexity)
        old_ctx = self._ctx_scores.get(ctx_key, 0.5)
        self._ctx_scores[ctx_key] = self.EMA_ALPHA * signal + (1.0 - self.EMA_ALPHA) * old_ctx
        sk_key  = (agent_id, skill)
        old_sk  = self._skill_scores.get(sk_key, 0.5)
        self._skill_scores[sk_key] = self.EMA_ALPHA * signal + (1.0 - self.EMA_ALPHA) * old_sk

    def get(self, agent_id: str, skill: str, tier: str = "standard", complexity: str = "medium") -> float:
        ctx = self._ctx_scores.get((agent_id, skill, tier, complexity))
        if ctx is not None:
            return ctx
        return self._skill_scores.get((agent_id, skill), 0.5)

    def get_skill(self, agent_id: str, skill: str) -> float:
        return self._skill_scores.get((agent_id, skill), 0.5)

    def print_affinity_table(self) -> None:
        if not self._skill_scores:
            print("  [ContextualAffinityModel] No affinity data recorded.")
            return
        print("\n── Agent Affinity Table ──────────────────────────────────────────")
        print(f"  {'Agent':<20}  {'Skill':<14}  {'Affinity':>8}  {'vs baseline':>12}")
        print("  " + "─" * 58)
        for (agent_id, skill), score in sorted(self._skill_scores.items()):
            delta = score - 0.5
            bar   = ("▲" if delta > 0 else "▼") * min(5, int(abs(delta) * 20))
            print(f"  {agent_id:<20}  {skill:<14}  {score:>8.4f}  {delta:>+10.4f}  {bar}")

    def print_contextual_report(self) -> None:
        interesting = {k: v for k, v in self._ctx_scores.items() if abs(v - 0.5) > 0.05}
        if not interesting:
            print("  [ContextualAffinityModel] No significant contextual differentiation yet.")
            return
        print("\n── Contextual Affinity (significant cells) ────────────────────")
        print(f"  {'Agent':<12}  {'Skill':<12}  {'Tier':<10}  {'Complexity':<10}  {'Score':>8}")
        for (agent, skill, tier, cplx), score in sorted(interesting.items()):
            print(f"  {agent:<12}  {skill:<12}  {tier:<10}  {cplx:<10}  {score:>8.4f}")


class AgentAffinityModel(ContextualAffinityModel):
    """Backward-compatible alias."""
    pass


# ---------------------------------------------------------------------------
# FatigueTracker
# ---------------------------------------------------------------------------

class FatigueTracker:
    """Tracks per-agent fatigue and burnout risk for the ML routing layer.

    When a realism engine is attached it delegates to that engine's per-agent
    state; otherwise it maintains its own accumulation. reset() clears all
    per-agent state and must be called at the start of every feedback epoch so
    fatigue does not carry over and saturate every agent at 1.0.
    """

    def __init__(
        self,
        accumulation_rate: float = 0.0015,
        recovery_rate:     float = 0.05,
        persistence:       float = 0.20,
        burnout_threshold: float = 0.70,
        burnout_slope:     float = 10.0,
    ) -> None:
        self._acc_rate  = accumulation_rate
        self._rec_rate  = recovery_rate
        self._persist   = persistence
        self._threshold = burnout_threshold
        self._slope     = burnout_slope
        self._fatigue:    Dict[str, float] = {}
        self._peak:       Dict[str, float] = {}
        self._total_work: Dict[str, float] = {}
        self._realism_engine: Optional[Any] = None

    def reset(self) -> None:
        """Clear all per-agent accumulated state between epochs."""
        self._fatigue.clear()
        self._peak.clear()
        self._total_work.clear()

    def set_realism_engine(self, engine: Any) -> None:
        self._realism_engine = engine

    def get_fatigue(self, agent_id: str) -> float:
        if self._realism_engine is not None:
            state = self._realism_engine.realism.get_state(agent_id)
            if state is not None:
                return state.fatigue.level
        return self._fatigue.get(agent_id, 0.0)

    def get_burnout_risk(self, agent_id: str) -> float:
        f = self.get_fatigue(agent_id)
        return float(1.0 / (1.0 + math.exp(-self._slope * (f - self._threshold))))

    def accumulate(self, agent_id: str, minutes: float, utilisation: float = 1.0) -> float:
        if self._realism_engine is not None:
            return self.get_fatigue(agent_id)
        old_f = self._fatigue.get(agent_id, 0.0)
        delta = self._acc_rate * minutes * (1.0 + 0.5 * utilisation)
        new_f = min(1.0, old_f + delta)
        self._fatigue[agent_id]    = new_f
        self._peak[agent_id]       = max(self._peak.get(agent_id, 0.0), new_f)
        self._total_work[agent_id] = self._total_work.get(agent_id, 0.0) + minutes
        return new_f

    def recover(self, agent_id: str, break_minutes: float) -> float:
        if self._realism_engine is not None:
            return self.get_fatigue(agent_id)
        old_f = self._fatigue.get(agent_id, 0.0)
        new_f = old_f * math.exp(-self._rec_rate * break_minutes)
        self._fatigue[agent_id] = max(0.0, new_f)
        return self._fatigue[agent_id]

    def end_of_shift(self, agent_id: str) -> float:
        end_f  = self.get_fatigue(agent_id)
        next_f = max(0.05, end_f * self._persist)
        if self._realism_engine is None:
            self._fatigue[agent_id] = next_f
        return next_f

    def aht_multiplier(self, agent_id: str) -> float:
        f = self.get_fatigue(agent_id)
        return 1.0 + 0.30 * f

    def print_report(self) -> None:
        print("\n── Fatigue Tracker ─────────────────────────────────────────────")
        if self._realism_engine is not None:
            print("  [Delegating to HumanRealisticsEngine — see realism report]")
            return
        if not self._fatigue:
            print("  [FatigueTracker] No data yet.")
            return
        print(f"  {'Agent':<12}  {'Fatigue':>8}  {'Peak':>8}  {'Burnout%':>10}  {'WorkMins':>10}")
        for aid in sorted(self._fatigue):
            print(
                f"  {aid:<12}  {self._fatigue[aid]:>8.3f}  "
                f"{self._peak.get(aid,0):>8.3f}  "
                f"{self.get_burnout_risk(aid):>9.1%}  "
                f"{self._total_work.get(aid,0):>10.1f}"
            )


# ===========================================================================
# PART 2 — ROUTING AND RL HOOKS
# ===========================================================================

class RLStateBuilder:
    def __init__(self, skills: List[str], max_agents: int = 20) -> None:
        self._skills     = skills
        self._max_agents = max_agents
        self.state_dim   = 6 + 3 * len(skills)

    def build(self, sla_rolling: float, shift_progress: float, mean_fatigue: float,
              mean_burnout: float, abn_rate_rolling: float, queue_depths: Dict[str, int],
              agents_available: Dict[str, int], utilizations: Dict[str, float]) -> np.ndarray:
        qd = [min(1.0, queue_depths.get(s, 0) / 10.0)    for s in self._skills]
        av = [min(1.0, agents_available.get(s, 0) / max(1, self._max_agents)) for s in self._skills]
        ut = [min(1.0, utilizations.get(s, 0.0))          for s in self._skills]
        return np.array([
            float(np.clip(sla_rolling,      0.0, 1.0)),
            float(np.clip(shift_progress,   0.0, 1.0)),
            float(np.clip(mean_fatigue,     0.0, 1.0)),
            float(np.clip(mean_burnout,     0.0, 1.0)),
            float(np.clip(abn_rate_rolling, 0.0, 1.0)),
            float(np.clip(max(qd) if qd else 0.0, 0.0, 1.0)),
            *qd, *av, *ut,
        ], dtype=np.float32)

    @staticmethod
    def compute_reward(sla_met: bool, fcr: float, burnout_risk: float, n_abandoned: int) -> float:
        return float(sla_met) + 0.5 * fcr - burnout_risk - float(n_abandoned)


# ---------------------------------------------------------------------------
# ML Router
# ---------------------------------------------------------------------------

class MLRouter(Router):
    # Relative weights of the five ML sub-signals (sum to 1.0); these form the
    # ML mixture only.
    _W_CSAT    = 0.30
    _W_FCR     = 0.20
    _W_ABANDON = 0.20
    _W_AFFIN   = 0.20
    _W_FATIGUE = 0.10
    # How much the ML mixture displaces the base heuristic score. Must be < 1.0
    # so the base composite still contributes (previously this was the *sum* of
    # the weights above, i.e. 1.0, which silently zeroed out the base score).
    ML_BLEND   = 0.60

    def __init__(
        self,
        agent_pool:      List[Agent],
        config:          SimulationConfig,
        weights:         Optional[RouterScoreWeights],
        csat_model:      Optional[CSATPredictionModel],
        abandon_model:   Optional[AbandonmentRiskModel],
        affinity_model:  ContextualAffinityModel,
        fcr_model:       Optional[FCRPredictionModel] = None,
        fatigue_tracker: Optional[FatigueTracker]    = None,
    ) -> None:
        super().__init__(agent_pool, config, weights)
        self._csat_model    = csat_model
        self._abandon_model = abandon_model
        self._affinity      = affinity_model
        self._fcr_model     = fcr_model
        self._fatigue       = fatigue_tracker

    def notify_call_ended(self, agent_id: str, csat_raw: float, handle_minutes: float,
                          is_repeat: bool, skill: str = "", tier: str = "standard",
                          queue_depth: int = 0) -> None:
        super().notify_call_ended(agent_id, csat_raw, handle_minutes, is_repeat, skill)
        if skill:
            self._affinity.update(agent_id, skill, csat_raw, handle_minutes, is_repeat,
                                  tier=tier, queue_depth=queue_depth)
        if self._fatigue is not None:
            self._fatigue.accumulate(agent_id, handle_minutes, utilisation=0.9)

    def _composite_score(self, agent: Agent, call: Call, skill_resources: Dict[str, Any]) -> float:
        # NOTE: skill_resources is keyed by *skill* (see Router._compute_signals
        # and the engine). The previous version looked things up by agent_id,
        # which never matched, so est_wait and queue_depth were dead defaults.
        base_score = super()._composite_score(agent, call, skill_resources)
        ml_score   = 0.0
        st         = self.tracker.get(agent.agent_id)
        resource   = skill_resources.get(call.skill)
        est_wait   = self.estimated_wait(resource) if resource is not None else 5.0
        queue_depth = (
            len(resource.queue)
            if resource is not None and hasattr(resource, "queue")
            else 0
        )

        if self._csat_model is not None and st.calls_completed >= 2:
            csat_features = np.array([
                _tier_to_int(call.customer_type), _skill_to_int(call.skill),
                float(call.is_repeat), st.ema_csat, st.ema_handle_time,
                st.ema_resolution, _exp_to_int(agent.experience), float(st.active_calls),
            ])
            ml_score += self._W_CSAT * self._csat_model.predict_proba_positive(csat_features)

        if self._fcr_model is not None and st.calls_completed >= 2:
            fcr_features = np.array([
                _skill_to_int(call.skill), _tier_to_int(call.customer_type),
                float(call.is_repeat), st.ema_resolution, st.ema_handle_time,
                _exp_to_int(agent.experience), float(_complexity_to_int(st.ema_handle_time)),
            ])
            ml_score += self._W_FCR * self._fcr_model.predict_fcr_prob(fcr_features)

        if self._abandon_model is not None:
            abn_features = np.array([
                est_wait, _tier_to_int(call.customer_type), _skill_to_int(call.skill),
                float(call.is_repeat), float(queue_depth), call.arrival_time,
            ])
            abn_prob    = self._abandon_model.predict_abandon_prob(
                abn_features, tier=call.customer_type, wait_minutes=est_wait,
            )
            speed_signal = max(0.0, 1.0 - st.ema_handle_time / 15.0)
            ml_score    += self._W_ABANDON * speed_signal * abn_prob

        affinity = self._affinity.get(
            agent.agent_id, call.skill,
            tier      =getattr(call, "customer_type", "standard"),
            complexity=_complexity_bucket(st.ema_handle_time),
        )
        ml_score += self._W_AFFIN * affinity

        if self._fatigue is not None:
            burnout_risk = self._fatigue.get_burnout_risk(agent.agent_id)
            ml_score    -= self._W_FATIGUE * burnout_risk

        return (1.0 - self.ML_BLEND) * base_score + self.ML_BLEND * ml_score


class MLSimulationEngine(SimulationEngine):
    def __init__(
        self,
        config:         SimulationConfig,
        router_factory: Optional[Callable] = None,
        weights:        Optional[RouterScoreWeights] = None,
    ) -> None:
        self._router_factory   = router_factory
        self._override_weights = weights
        super().__init__(config, weights)
        if router_factory is not None:
            self.router = router_factory(self.agents, config, weights)
            if hasattr(self.router, "_fcr_model"):
                self._fcr_model = self.router._fcr_model


# ---------------------------------------------------------------------------
# ML Model Registry
# ---------------------------------------------------------------------------

class MLModelRegistry:
    def __init__(self) -> None:
        self.csat_predictor     = CSATPredictionModel()
        self.abandonment_model  = AbandonmentRiskModel()
        self.fcr_predictor      = FCRPredictionModel()
        self.arrival_forecaster = IntradayArrivalForecaster()
        self.affinity_model     = ContextualAffinityModel()
        self.fatigue_tracker    = FatigueTracker()
        self.rl_state_builder   = RLStateBuilder(
            skills=["billing", "technical", "general"], max_agents=20,
        )

    def warm_start(self) -> "MLModelRegistry":
        self.csat_predictor.warm_start()
        self.abandonment_model.warm_start()
        self.fcr_predictor.warm_start()
        self.arrival_forecaster.warm_start_synthetic()
        return self

    def make_router(self, agent_pool: List[Agent], config: SimulationConfig,
                    weights: Optional[RouterScoreWeights] = None) -> MLRouter:
        return MLRouter(
            agent_pool     =agent_pool, config=config, weights=weights,
            csat_model     =self.csat_predictor,
            abandon_model  =self.abandonment_model,
            affinity_model =self.affinity_model,
            fcr_model      =self.fcr_predictor,
            fatigue_tracker=self.fatigue_tracker,
        )

    def print_model_report(self) -> None:
        w = 72
        print(f"\n{'=' * w}")
        print(f"  ML MODEL REGISTRY REPORT")
        print(f"{'=' * w}")
        self.csat_predictor.print_report()
        self.abandonment_model.print_report()
        self.fcr_predictor.print_report()
        self.arrival_forecaster.print_report()
        n_sk = len(self.affinity_model._skill_scores)
        n_cx = len(self.affinity_model._ctx_scores)
        print(f"  ContextualAffinityModel  [{n_sk} skill pairs, {n_cx} contextual cells]")
        self.fatigue_tracker.print_report()
        print(f"  RLStateBuilder           [state_dim={self.rl_state_builder.state_dim}]")
        print(f"{'=' * w}\n")


# ===========================================================================
# PART 3 — ADAPTIVE FEEDBACK LOOP
# ===========================================================================

@dataclass
class FeedbackConfig:
    enabled:                   bool  = True
    n_epochs:                  int   = 5
    accumulation_strategy:     Literal["cumulative", "rolling"] = "cumulative"
    rolling_window:            int   = 3
    retrain_csat_model:        bool  = True
    retrain_arrival_model:     bool  = False
    retrain_abandonment_model: bool  = True
    retrain_fcr_model:         bool  = True
    online_partial_fit:        bool  = False
    convergence_metric:        Literal["sla", "csat", "abandonment", "fcr"] = "csat"
    convergence_delta:         float = 0.005
    convergence_patience:      int   = 3
    drift_detection:           bool  = True
    verbose:                   bool  = True
    base_seed:                 int   = 42


@dataclass
class RunRecord:
    epoch:           int
    call_records:    List[_CallRecord]
    agent_stats:     Optional[List]
    kpi:             Dict[str, float]
    elapsed_seconds: float = 0.0

    def metric(self, name: str) -> float:
        return self.kpi.get(name, 0.0)


class DataCollector:
    @staticmethod
    def csat_dataset(run: RunRecord) -> Tuple[np.ndarray, np.ndarray]:
        X_rows, y_rows = [], []
        rng = np.random.default_rng(seed=run.epoch + 1)
        noise_scale = 0.02
        for rec in run.call_records:
            csat_norm = (rec.csat_raw - 1.0) / 4.0
            row = [
                _tier_to_int(rec.customer_type), _skill_to_int(rec.skill),
                float(rec.is_repeat),
                float(np.clip(csat_norm + rng.normal(0, noise_scale), 0.0, 1.0)),
                float(max(0.5, rec.handle_minutes + rng.normal(0, noise_scale * 5))),
                0.0 if rec.is_repeat else 1.0, 1, 1,
            ]
            X_rows.append(row)
            y_rows.append(int(rec.csat_raw >= 4.0))
        if not X_rows:
            return np.empty((0, 8)), np.empty(0, dtype=int)
        return np.array(X_rows, dtype=float), np.array(y_rows, dtype=int)

    @staticmethod
    def abandonment_dataset(run: RunRecord, sim_duration: float = 480.0) -> Tuple[np.ndarray, np.ndarray]:
        X_rows, y_rows = [], []
        rng = np.random.default_rng(seed=run.epoch + 100)
        for rec in run.call_records:
            window_depth = int(rec.arrival_time / 10.0)
            X_rows.append([
                rec.wait_minutes + rng.normal(0, 0.03),
                _tier_to_int(rec.customer_type), _skill_to_int(rec.skill),
                float(rec.is_repeat), float(window_depth), rec.arrival_time,
            ])
            y_rows.append(0)
        abn_rate = run.kpi.get("abandonment", 0.0)
        n_ans    = len(run.call_records)
        n_abn    = max(30, int(round(abn_rate / max(1 - abn_rate, 1e-6) * n_ans)))
        if n_ans > 0:
            idx = rng.integers(0, n_ans, size=n_abn)
            for i in idx:
                rec = run.call_records[i]
                X_rows.append([
                    float(max(0.0, rng.exponential(3.0))), rng.integers(0, 2),
                    _skill_to_int(rec.skill), float(rec.is_repeat),
                    float(rng.integers(3, 10)), rec.arrival_time,
                ])
                y_rows.append(1)
        if not X_rows:
            return np.empty((0, 6)), np.empty(0, dtype=int)
        return np.array(X_rows, dtype=float), np.array(y_rows, dtype=int)

    @staticmethod
    def survival_fit_data(run: RunRecord) -> Tuple[List[float], List[int], List[str]]:
        wait_times, flags, tiers = [], [], []
        rng = np.random.default_rng(seed=run.epoch + 200)
        for rec in run.call_records:
            wait_times.append(rec.wait_minutes); flags.append(0); tiers.append(rec.customer_type)
        abn_rate = run.kpi.get("abandonment", 0.0)
        n_ans    = len(run.call_records)
        n_abn    = max(10, int(round(abn_rate / max(1 - abn_rate, 1e-6) * n_ans)))
        if n_ans > 0:
            tier_choices = ["standard", "premium", "vip"]
            for _ in range(n_abn):
                wait_times.append(float(rng.exponential(3.0)))
                flags.append(1)
                tiers.append(str(rng.choice(tier_choices)))
        return wait_times, flags, tiers

    @staticmethod
    def fcr_dataset(run: RunRecord) -> Tuple[np.ndarray, np.ndarray]:
        X_rows, y_rows = [], []
        rng = np.random.default_rng(seed=run.epoch + 300)
        for rec in run.call_records:
            X_rows.append([
                _skill_to_int(rec.skill), _tier_to_int(rec.customer_type),
                float(rec.is_repeat), max(0.0, min(1.0, (rec.csat_raw - 1.0) / 4.0)),
                float(rec.handle_minutes), 1, float(_complexity_to_int(rec.handle_minutes)),
            ])
            y_rows.append(int(not rec.is_repeat))
        if not X_rows:
            return np.empty((0, 7)), np.empty(0, dtype=int)
        return np.array(X_rows, dtype=float), np.array(y_rows, dtype=int)

    @staticmethod
    def arrival_dataset(run: RunRecord, n_intervals: int = 32, sim_duration: float = 480.0) -> np.ndarray:
        interval_min = sim_duration / n_intervals
        counts       = np.zeros(n_intervals)
        for rec in run.call_records:
            idx = min(n_intervals - 1, int(rec.arrival_time / interval_min))
            counts[idx] += 1
        return counts / interval_min


# ---------------------------------------------------------------------------
# ModelUpdater
# ---------------------------------------------------------------------------

class ModelUpdater:
    def __init__(self, registry: MLModelRegistry, config: FeedbackConfig) -> None:
        self._registry = registry
        self._cfg      = config

    def update(self, csat_X: Optional[np.ndarray], csat_y: Optional[np.ndarray],
               abandon_X: Optional[np.ndarray], abandon_y: Optional[np.ndarray],
               fcr_X: Optional[np.ndarray] = None, fcr_y: Optional[np.ndarray] = None,
               survival: Optional[Tuple] = None, arrival: Optional[np.ndarray] = None,
               drift_triggered: bool = False, epoch_seed: int = 42) -> Dict[str, bool]:
        updated: Dict[str, bool] = {
            "csat": False, "abandonment": False, "fcr": False,
            "arrival": False, "survival": False,
        }
        if (self._cfg.retrain_csat_model and csat_X is not None
                and len(csat_X) >= 20 and len(np.unique(csat_y)) > 1):
            _reset_numpy_seed(epoch_seed)
            if self._cfg.online_partial_fit and not drift_triggered:
                self._registry.csat_predictor.partial_fit(csat_X, csat_y)
            else:
                self._registry.csat_predictor.fit(csat_X, csat_y)
            updated["csat"] = True

        if (self._cfg.retrain_abandonment_model and abandon_X is not None
                and len(abandon_X) >= 20 and len(np.unique(abandon_y)) > 1):
            _reset_numpy_seed(epoch_seed + 1000)
            if self._cfg.online_partial_fit and not drift_triggered:
                self._registry.abandonment_model.partial_fit(abandon_X, abandon_y)
            else:
                self._registry.abandonment_model.fit(abandon_X, abandon_y)
            updated["abandonment"] = True

        if survival is not None:
            wait_times, flags, tiers = survival
            self._registry.abandonment_model.fit_survival(wait_times, flags, tiers)
            updated["survival"] = True

        if (self._cfg.retrain_fcr_model and fcr_X is not None
                and len(fcr_X) >= 20 and len(np.unique(fcr_y)) > 1):
            _reset_numpy_seed(epoch_seed + 2000)
            self._registry.fcr_predictor.fit(fcr_X, fcr_y)
            updated["fcr"] = True

        if self._cfg.retrain_arrival_model and arrival is not None:
            _reset_numpy_seed(epoch_seed + 3000)
            self._registry.arrival_forecaster.append_shift(arrival)
            if len(self._registry.arrival_forecaster._history) >= 3:
                self._registry.arrival_forecaster.fit(self._registry.arrival_forecaster._history)
            updated["arrival"] = True

        return updated


# ---------------------------------------------------------------------------
# ConvergenceTracker
# ---------------------------------------------------------------------------

class ConvergenceTracker:
    _INVERT = {"abandonment"}

    def __init__(self, metric: str, delta: float, patience: int) -> None:
        self._metric   = metric
        self._delta    = delta
        self._patience = patience
        self._history: List[float] = []
        self._no_improve = 0
        self._best: float = float("-inf")
        self._ph = PageHinkleyDrift(delta=delta / 2, lambda_=patience * 10.0)

    def record(self, value: float) -> None:
        effective = -value if self._metric in self._INVERT else value
        self._history.append(effective)
        drift = self._ph.update(effective)
        if drift:
            self._no_improve = 0; self._best = effective; return
        if effective > self._best + self._delta:
            self._best = effective; self._no_improve = 0
        else:
            self._no_improve += 1

    def has_converged(self) -> bool:
        return len(self._history) >= self._patience and self._no_improve >= self._patience

    @property
    def drift_detected(self) -> bool:
        return self._ph.drift_detected

    @property
    def history(self) -> List[float]:
        return [-v if self._metric in self._INVERT else v for v in self._history]


# ---------------------------------------------------------------------------
# FeedbackLoopRunner
# ---------------------------------------------------------------------------

class FeedbackLoopRunner:
    """Runs the adaptive simulate -> collect -> retrain loop over N epochs.

    Each epoch reseeds the global RNGs and resets the shared fatigue tracker so
    runs are reproducible and fatigue does not accumulate across epochs.
    """

    def __init__(
        self,
        sim_config:      SimulationConfig,
        feedback_config: FeedbackConfig,
        registry:        MLModelRegistry,
        weights:         Optional[RouterScoreWeights] = None,
    ) -> None:
        self._sim_cfg     = sim_config
        self._fb_cfg      = feedback_config
        self._reg         = registry
        self._weights     = weights
        self._updater     = ModelUpdater(registry, feedback_config)
        self._convergence = ConvergenceTracker(
            metric  =feedback_config.convergence_metric,
            delta   =feedback_config.convergence_delta,
            patience=feedback_config.convergence_patience,
        )
        self._pool: Dict[str, List[np.ndarray]] = {
            "csat_X": [], "csat_y": [],
            "abn_X":  [], "abn_y":  [],
            "fcr_X":  [], "fcr_y":  [],
        }

    def run(self) -> List[RunRecord]:
        cfg     = self._fb_cfg
        history: List[RunRecord] = []

        if not cfg.enabled:
            if cfg.verbose:
                print("[FeedbackLoop] enabled=False — single baseline pass.")
            history.append(self._run_epoch(epoch=0))
            return history

        for epoch in range(cfg.n_epochs):
            epoch_seed = cfg.base_seed + epoch * 17

            if cfg.verbose:
                print(f"\n{'=' * 60}")
                print(f"  [FeedbackLoop] Epoch {epoch + 1}/{cfg.n_epochs}  (seed={epoch_seed})")
                print(f"{'=' * 60}")

            record = self._run_epoch(epoch=epoch)
            history.append(record)

            if cfg.verbose:
                self._print_epoch_summary(record)

            self._accumulate(record)
            metric_val = record.metric(cfg.convergence_metric)
            self._convergence.record(metric_val)

            if epoch > 0 and self._convergence.has_converged():
                if cfg.verbose:
                    print(f"  [FeedbackLoop] Convergence after epoch {epoch + 1}. Stopping.")
                break

            if epoch < cfg.n_epochs - 1:
                drift   = self._convergence.drift_detected if cfg.drift_detection else False
                updated = self._updater.update(
                    csat_X   =self._pooled("csat_X"), csat_y=self._pooled("csat_y"),
                    abandon_X=self._pooled("abn_X"),  abandon_y=self._pooled("abn_y"),
                    fcr_X    =self._pooled("fcr_X"),  fcr_y=self._pooled("fcr_y"),
                    survival =DataCollector.survival_fit_data(record),
                    arrival  =DataCollector.arrival_dataset(record),
                    drift_triggered=drift,
                    epoch_seed=epoch_seed,
                )
                if cfg.verbose:
                    trained   = [k for k, v in updated.items() if v]
                    drift_tag = "  [DRIFT — full refit]" if drift else ""
                    print(f"  [FeedbackLoop] Retrained: {trained or 'none'}{drift_tag}")

        return history

    def _run_epoch(self, epoch: int) -> RunRecord:
        t0         = time.perf_counter()
        epoch_seed = self._fb_cfg.base_seed + epoch * 17

        # Reset BOTH global RNGs before building the engine (the engine itself
        # also seeds its own generator from epoch_cfg.random_seed).
        random.seed(epoch_seed)
        np.random.seed(epoch_seed & 0xFFFFFFFF)

        # Reset fatigue tracker so each epoch starts with zero accumulated work.
        if hasattr(self._reg, "fatigue_tracker"):
            self._reg.fatigue_tracker.reset()

        epoch_cfg             = copy.copy(self._sim_cfg)
        epoch_cfg.random_seed = epoch_seed

        engine = MLSimulationEngine(
            config        =epoch_cfg,
            router_factory=self._reg.make_router,
            weights       =self._weights,
        )

        # Attach fatigue tracker to engine if a realism layer is present.
        if hasattr(engine, "realism") and hasattr(self._reg, "fatigue_tracker"):
            self._reg.fatigue_tracker.set_realism_engine(engine)

        engine.run()
        elapsed = time.perf_counter() - t0

        agent_stats = None
        if _COST_AVAILABLE:
            try:
                agent_stats = AgentUtilizationCollector(engine).collect()
            except Exception:
                pass

        return RunRecord(
            epoch          =epoch,
            call_records   =list(engine.kpi._records),
            agent_stats    =agent_stats,
            kpi            =self._extract_kpi(engine),
            elapsed_seconds=elapsed,
        )

    @staticmethod
    def _extract_kpi(engine: MLSimulationEngine) -> Dict[str, float]:
        return {
            "sla":         engine.kpi.sla_percentage(),
            "abandonment": engine.kpi.abandonment_rate(),
            "csat":        engine.kpi.average_csat(),
            "asa":         engine.kpi.average_speed_of_answer(),
            "aht":         engine.kpi.average_handle_time(),
            "fcr":         engine.kpi.first_call_resolution(),
            "calls":       float(engine.kpi.total_calls()),
        }

    def _accumulate(self, run: RunRecord) -> None:
        X_csat, y_csat = DataCollector.csat_dataset(run)
        X_abn,  y_abn  = DataCollector.abandonment_dataset(run, self._sim_cfg.sim_duration_minutes)
        X_fcr,  y_fcr  = DataCollector.fcr_dataset(run)
        for key, arr in [
            ("csat_X", X_csat), ("csat_y", y_csat),
            ("abn_X",  X_abn),  ("abn_y",  y_abn),
            ("fcr_X",  X_fcr),  ("fcr_y",  y_fcr),
        ]:
            if len(arr) > 0:
                self._pool[key].append(arr)
        if self._fb_cfg.accumulation_strategy == "rolling":
            w = self._fb_cfg.rolling_window
            for key in self._pool:
                self._pool[key] = self._pool[key][-w:]

    def _pooled(self, key: str) -> Optional[np.ndarray]:
        arrays = self._pool.get(key, [])
        if not arrays:
            return None
        stacker = np.vstack if arrays[0].ndim == 2 else np.concatenate
        return stacker(arrays)

    @staticmethod
    def _print_epoch_summary(run: RunRecord) -> None:
        k = run.kpi
        print(
            f"  SLA={k['sla']:.1%}  Abandon={k['abandonment']:.1%}  "
            f"CSAT={k['csat']:.3f}  ASA={k['asa']:.2f}m  "
            f"FCR={k['fcr']:.1%}  Calls={k['calls']:.0f}  [{run.elapsed_seconds:.1f}s]"
        )


# ---------------------------------------------------------------------------
# FeedbackLoopReport
# ---------------------------------------------------------------------------

class FeedbackLoopReport:
    _METRICS = [
        ("sla",         "SLA %",      "{:.1%}", True),
        ("abandonment", "Abandon %",  "{:.1%}", False),
        ("csat",        "Avg CSAT",   "{:.3f}", True),
        ("asa",         "ASA (min)",  "{:.2f}", False),
        ("fcr",         "FCR %",      "{:.1%}", True),
        ("calls",       "Calls Hdld", "{:.0f}", True),
    ]

    def __init__(self, history: List[RunRecord], config: FeedbackConfig) -> None:
        self._history = history
        self._cfg     = config

    def print_report(self) -> None:
        w   = 80
        bar = "=" * w
        print(f"\n{bar}")
        print(f"  ADAPTIVE ML FEEDBACK LOOP -- EPOCH PROGRESSION REPORT")
        print(
            f"  Epochs run: {len(self._history)}  |  "
            f"Strategy: {self._cfg.accumulation_strategy}  |  "
            f"Convergence metric: {self._cfg.convergence_metric}"
        )
        print(
            f"  Seed per epoch: base={self._cfg.base_seed}, step=17  |  "
            f"Drift detection: {self._cfg.drift_detection}  |  "
            f"Reproducible: YES"
        )
        print(f"{bar}")

        header = f"  {'Epoch':>6}"
        for _, label, _, _ in self._METRICS:
            header += f"  {label:>12}"
        header += f"  {'Time(s)':>8}"
        print(header)
        print(f"  {'-' * (w - 4)}")

        best_epochs: Dict[str, int] = {}
        for key, _, _, higher_better in self._METRICS:
            values = [r.metric(key) for r in self._history]
            best_epochs[key] = int(np.argmax(values)) if higher_better else int(np.argmin(values))

        for run in self._history:
            row = f"  {run.epoch + 1:>6}"
            for key, _, fmt, _ in self._METRICS:
                val    = run.metric(key)
                marker = "*" if run.epoch == best_epochs[key] else " "
                row   += f"  {marker}{fmt.format(val):>11}"
            row += f"  {run.elapsed_seconds:>8.1f}"
            print(row)

        print(f"  {'-' * (w - 4)}")
        print(f"  * = best epoch for that metric")

        if len(self._history) >= 2:
            print(f"\n  Delta: Epoch {len(self._history)} vs Epoch 1")
            first = self._history[0]; last = self._history[-1]
            for key, label, fmt, higher_better in self._METRICS:
                v0      = first.metric(key); v1 = last.metric(key); delta = v1 - v0
                sign    = "+" if delta >= 0 else ""
                improved = (delta > 0) == higher_better
                tag      = "✓ improved" if improved else "✗ regressed"
                print(
                    f"  {label:<14}  {fmt.format(v0)} → {fmt.format(v1)}  "
                    f"({sign}{fmt.format(delta)})  {tag}"
                )

        print(f"\n{bar}")
        print(f"  Training data accumulated:")
        print(f"    Calls per epoch (approx):   ~{len(self._history[0].call_records)}")
        print(f"    Convergence patience:        {self._cfg.convergence_patience} epochs")
        print(
            f"    Convergence delta ({self._cfg.convergence_metric}): "
            f"{self._cfg.convergence_delta:.4f}"
        )
        print(f"{bar}\n")