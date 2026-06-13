"""
eda.py  —  exploratory-data-analysis / data-quality layer
=========================================================

Sits between the `DecisionLog` (which collects real (features, label) rows during
a simulation epoch) and the predictors. Before any data enters a model, this
layer inspects it and produces a structured `DatasetDiagnostics` report:

  * shape, per-feature summary stats, NaN/inf checks
  * degenerate (zero-variance) features that contribute nothing
  * multicollinearity (highly correlated feature pairs)
  * LEAKAGE GUARD — any feature whose correlation with the label is implausibly
    high (this is exactly the failure mode of the old design, where the CSAT
    label leaked in as a feature; the guard would now flag it)
  * class balance / positive rate, with a note on AUC reliability
  * feature drift versus a running reference from previous epochs
  * a single `trainable` verdict + human-readable warnings

The feedback loop calls `EDALayer.analyze(...)` on each dataset and uses the
verdict to gate training, so bad data is caught and reported instead of silently
degrading a model.

Depends only on numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ===========================================================================
# Report dataclasses
# ===========================================================================

@dataclass
class FeatureStats:
    name:          str
    mean:          float
    std:           float
    minimum:       float
    maximum:       float
    n_unique:      int
    zero_variance: bool
    target_corr:   float           # point-biserial corr with the label


@dataclass
class DatasetDiagnostics:
    target_name:        str
    n_rows:             int
    n_features:         int
    pos_rate:           float
    n_missing:          int
    feature_stats:      List[FeatureStats] = field(default_factory=list)
    degenerate:         List[str]                       = field(default_factory=list)
    collinear_pairs:    List[Tuple[str, str, float]]    = field(default_factory=list)
    leakage_suspects:   List[Tuple[str, float]]         = field(default_factory=list)
    drift_score:        Optional[float]                 = None
    drifted_features:   List[Tuple[str, float]]         = field(default_factory=list)
    warnings:           List[str]                       = field(default_factory=list)
    trainable:          bool                            = True

    def summary(self) -> str:
        head = (f"[EDA:{self.target_name}] n={self.n_rows} feats={self.n_features} "
                f"pos_rate={self.pos_rate:.1%} "
                f"trainable={'yes' if self.trainable else 'NO'}")
        drift = "" if self.drift_score is None else f" drift={self.drift_score:.3f}"
        lines = [head + drift]
        for w in self.warnings:
            lines.append(f"    ! {w}")
        return "\n".join(lines)


# ===========================================================================
# EDA layer
# ===========================================================================

class EDALayer:
    """Stateless analysis + a small running reference for cross-epoch drift."""

    def __init__(self,
                 min_rows: int = 40,
                 leakage_corr: float = 0.92,
                 collinear_corr: float = 0.95,
                 imbalance_rate: float = 0.05,
                 drift_threshold: float = 0.25,
                 reference_decay: float = 0.5) -> None:
        self._min_rows        = min_rows
        self._leak            = leakage_corr
        self._collinear       = collinear_corr
        self._imbalance       = imbalance_rate
        self._drift_threshold = drift_threshold
        self._decay           = reference_decay
        self._ref_mean: Optional[np.ndarray] = None
        self._ref_std:  Optional[np.ndarray] = None

    # -- main entry point ---------------------------------------------------

    def analyze(self, X: np.ndarray, y: np.ndarray,
                feature_names: Sequence[str],
                target_name: str = "target",
                update_reference: bool = True) -> DatasetDiagnostics:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        n_rows = int(X.shape[0])
        n_feat = int(X.shape[1]) if X.ndim == 2 else 0

        diag = DatasetDiagnostics(
            target_name=target_name, n_rows=n_rows, n_features=n_feat,
            pos_rate=float(np.mean(y)) if n_rows else 0.0,
            n_missing=int(np.count_nonzero(~np.isfinite(X))) if n_rows else 0,
        )

        if n_rows == 0 or n_feat == 0:
            diag.trainable = False
            diag.warnings.append("empty dataset")
            return diag

        # ---- NaN / inf -----------------------------------------------------
        if diag.n_missing > 0:
            diag.warnings.append(f"{diag.n_missing} non-finite values present")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # ---- per-feature stats + leakage guard -----------------------------
        means = X.mean(axis=0)
        stds  = X.std(axis=0)
        y_centered = y - y.mean()
        y_var = float(np.dot(y_centered, y_centered))

        for j in range(n_feat):
            col   = X[:, j]
            zvar  = bool(stds[j] < 1e-9)
            tcorr = 0.0
            if not zvar and y_var > 0:
                c = col - means[j]
                denom = float(np.sqrt(np.dot(c, c) * y_var))
                tcorr = float(np.dot(c, y_centered) / denom) if denom > 0 else 0.0
            name = feature_names[j] if j < len(feature_names) else f"f{j}"
            diag.feature_stats.append(FeatureStats(
                name=name, mean=float(means[j]), std=float(stds[j]),
                minimum=float(col.min()), maximum=float(col.max()),
                n_unique=int(np.unique(col).size), zero_variance=zvar,
                target_corr=tcorr,
            ))
            if zvar:
                diag.degenerate.append(name)
            if abs(tcorr) >= self._leak:
                diag.leakage_suspects.append((name, tcorr))

        # ---- multicollinearity --------------------------------------------
        live = [j for j in range(n_feat) if stds[j] >= 1e-9]
        if len(live) >= 2:
            corr = np.corrcoef(X[:, live].T)
            for a in range(len(live)):
                for b in range(a + 1, len(live)):
                    r = corr[a, b]
                    if np.isfinite(r) and abs(r) >= self._collinear:
                        diag.collinear_pairs.append(
                            (feature_names[live[a]], feature_names[live[b]], float(r)))

        # ---- drift versus running reference -------------------------------
        if self._ref_mean is not None and self._ref_mean.shape == means.shape:
            denom = self._ref_std + 1e-9
            shift = np.abs(means - self._ref_mean) / denom
            diag.drift_score = float(np.mean(shift))
            for j in range(n_feat):
                if shift[j] >= self._drift_threshold:
                    nm = feature_names[j] if j < len(feature_names) else f"f{j}"
                    diag.drifted_features.append((nm, float(shift[j])))

        if update_reference:
            self._update_reference(means, stds)

        # ---- warnings + trainable verdict ---------------------------------
        if n_rows < self._min_rows:
            diag.warnings.append(f"only {n_rows} rows (< {self._min_rows}); fit may be unstable")
            diag.trainable = False
        if len(np.unique(y)) < 2:
            diag.warnings.append("single-class label; cannot train a classifier")
            diag.trainable = False
        if min(diag.pos_rate, 1.0 - diag.pos_rate) < self._imbalance and diag.trainable:
            diag.warnings.append(
                f"class imbalance (pos_rate={diag.pos_rate:.1%}); AUC ok, calibration suspect")
        if diag.degenerate:
            diag.warnings.append(f"{len(diag.degenerate)} zero-variance feature(s): "
                                 + ", ".join(diag.degenerate[:5]))
        if diag.leakage_suspects:
            names = ", ".join(f"{n}({c:+.2f})" for n, c in diag.leakage_suspects[:5])
            diag.warnings.append(f"possible target leakage: {names}")
        if diag.collinear_pairs:
            diag.warnings.append(f"{len(diag.collinear_pairs)} highly-collinear feature pair(s)")
        if diag.drifted_features:
            diag.warnings.append(f"{len(diag.drifted_features)} feature(s) drifted vs prior epoch")

        return diag

    def _update_reference(self, means: np.ndarray, stds: np.ndarray) -> None:
        if self._ref_mean is None:
            self._ref_mean, self._ref_std = means.copy(), stds.copy()
        else:
            d = self._decay
            self._ref_mean = d * self._ref_mean + (1 - d) * means
            self._ref_std  = d * self._ref_std + (1 - d) * stds

    def reset_reference(self) -> None:
        self._ref_mean = self._ref_std = None