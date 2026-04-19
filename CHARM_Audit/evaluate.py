"""HAB-event evaluation metrics.

Metrics implemented (README section 5):

    Categorical:  POD, FAR, CSI, F1, accuracy
    Probabilistic: Brier (with Murphy's reliability/resolution/uncertainty
                   decomposition), PR-AUC (the appropriate curve for rare
                   events; ROC-AUC is misleading), reliability table
    Threshold-tuning helper: best F1 / best CSI threshold by sweep
    Cost-sensitive helper:   POD at FAR <= 0.3   (literature convention)

Usage with the baselines:

    from evaluate import binary_event_report
    report = binary_event_report(y_true=y, p_pred=p, name="LightGBM/p_pda")
    print(report.as_table())

Continuous-chl-a metrics (log-RMSE, CRPS, FSS) are deferred until you
have the gridded short-horizon model -- they are not meaningful on the
station-level event head.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------
# Confusion matrix at a chosen threshold
# -----------------------------------------------------------------------
@dataclass
class ConfusionStats:
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def n(self) -> int: return self.tp + self.fp + self.fn + self.tn
    @property
    def pod(self) -> float:  # probability of detection (recall)
        d = self.tp + self.fn
        return self.tp / d if d else float("nan")
    @property
    def far(self) -> float:  # false-alarm ratio
        d = self.tp + self.fp
        return self.fp / d if d else float("nan")
    @property
    def csi(self) -> float:  # critical success index
        d = self.tp + self.fp + self.fn
        return self.tp / d if d else float("nan")
    @property
    def f1(self) -> float:
        d = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / d if d else float("nan")
    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else float("nan")
    @property
    def recall(self) -> float: return self.pod


def confusion_at(y_true: np.ndarray, p_pred: np.ndarray, threshold: float) -> ConfusionStats:
    """Confusion matrix at one threshold.

    NaN predictions are dropped before scoring (otherwise ``NaN >= thr``
    silently becomes a "predicted negative" and inflates TN counts).
    """
    y = np.asarray(y_true)
    p = np.asarray(p_pred, dtype=float)
    mask = ~(np.isnan(p) | np.isnan(y.astype(float, copy=False)))
    y, p = y[mask], p[mask]
    yh = (p >= threshold).astype(int)
    yt = y.astype(int)
    tp = int(((yh == 1) & (yt == 1)).sum())
    fp = int(((yh == 1) & (yt == 0)).sum())
    fn = int(((yh == 0) & (yt == 1)).sum())
    tn = int(((yh == 0) & (yt == 0)).sum())
    return ConfusionStats(threshold, tp, fp, fn, tn)


# -----------------------------------------------------------------------
# Brier with Murphy decomposition
# -----------------------------------------------------------------------
@dataclass
class BrierDecomp:
    brier: float
    reliability: float   # lower is better
    resolution: float    # higher is better
    uncertainty: float   # base-rate term


def brier_decomposition(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10) -> BrierDecomp:
    """Brier score with Murphy reliability/resolution/uncertainty decomposition.

    Predictions are clipped to ``[0, 1]`` before binning so the
    decomposition covers exactly the rows that contribute to the raw
    Brier (``reliability + uncertainty - resolution`` then equals
    ``brier`` to machine precision). Calibrators occasionally emit
    values just outside [0, 1]; without the clip those rows fell into
    bin -1 / bin n_bins and were silently dropped from the
    decomposition only.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p_pred, dtype=float), 0.0, 1.0)
    base = float(y.mean())
    uncertainty = base * (1 - base)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[-1] += 1e-9
    bins = np.digitize(p, edges) - 1
    rel = res = 0.0
    for k in range(n_bins):
        mask = bins == k
        n_k = mask.sum()
        if n_k == 0:
            continue
        p_k = p[mask].mean()
        o_k = y[mask].mean()
        w = n_k / len(y)
        rel += w * (p_k - o_k) ** 2
        res += w * (o_k - base) ** 2
    brier = float(np.mean((p - y) ** 2))
    return BrierDecomp(brier=brier, reliability=rel, resolution=res, uncertainty=uncertainty)


# -----------------------------------------------------------------------
# PR-AUC (rare events: README requires this, NOT ROC-AUC)
#
# Uses sklearn's ``average_precision_score`` -- the step-sum estimator
# Sigma_k (R_k - R_{k-1}) * P_k. The trapezoidal estimator we used to ship
# (np.trapz over the sorted precision-recall curve) consistently
# overestimates skill on rare-event problems by interpolating between PR
# points, implying achievable thresholds that don't exist; it also breaks
# down whenever the predictor produces tied probabilities (very common
# with isotonic-calibrated outputs). The step estimator matches the HAB /
# rare-event literature convention.
# -----------------------------------------------------------------------
def pr_auc(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    mask = ~(np.isnan(y) | np.isnan(p))
    y, p = y[mask], p[mask]
    if y.size == 0 or y.sum() == 0 or y.sum() == y.size:
        # all-negative *or* all-positive: average_precision_score is
        # undefined (or trivially 1.0) and not informative either way.
        return float("nan")
    return float(average_precision_score(y.astype(int), p))


# -----------------------------------------------------------------------
# Reliability table (the source for the README's reliability diagrams)
# -----------------------------------------------------------------------
def reliability_table(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 10,
    *,
    strategy: str = "uniform",
    min_bin_n: int = 1,
) -> pd.DataFrame:
    """Bin predictions and compute observed event rate per bin.

    ``strategy`` selects how the bin edges are chosen:

    - ``"uniform"`` (default): equal-width bins on ``[0, 1]``. Honest
      when predictions span the full unit interval.
    - ``"quantile"``: edges at predicted-probability quantiles, so
      every bin (modulo ties) holds approximately the same number of
      forecasts. Required for rare-event targets like ``p_pda`` whose
      predictions cluster near the base rate -- uniform binning puts
      ~all rows in bin 0 and the diagram looks flat.

    ``min_bin_n`` drops bins with fewer than that many forecasts before
    returning, so a single-row tail bin does not dominate the figure.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    mask = ~(np.isnan(y) | np.isnan(p))
    y, p = y[mask], p[mask]

    if strategy == "quantile" and p.size > 0:
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(p, qs))
        if edges.size < 2:
            # Predictions are constant; fall back to uniform so the
            # caller still gets a single-row table rather than an empty
            # DataFrame.
            edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges = edges.astype(float)
    edges[-1] = edges[-1] + 1e-9

    bins = np.digitize(p, edges) - 1
    rows = []
    for k in range(len(edges) - 1):
        m = bins == k
        n_k = int(m.sum())
        if n_k == 0:
            rows.append((float(edges[k]), float(edges[k+1]), 0, np.nan, np.nan))
            continue
        rows.append((
            float(edges[k]), float(edges[k+1]), n_k,
            float(p[m].mean()), float(y[m].mean()),
        ))
    out = pd.DataFrame(rows, columns=["bin_lo", "bin_hi", "n", "mean_p", "obs_rate"])
    if min_bin_n > 1:
        out = out[out["n"] >= min_bin_n].reset_index(drop=True)
    return out


# -----------------------------------------------------------------------
# Threshold tuning helpers
# -----------------------------------------------------------------------
def best_threshold(
    y_true: np.ndarray, p_pred: np.ndarray,
    *, by: str = "csi", grid: Sequence[float] | None = None,
) -> ConfusionStats:
    """Sweep thresholds and pick the one optimizing CSI / F1."""
    grid = grid or np.linspace(0.05, 0.95, 91)
    best, best_score = None, -np.inf
    for thr in grid:
        c = confusion_at(y_true, p_pred, thr)
        s = getattr(c, by)
        if np.isnan(s):
            continue
        if s > best_score:
            best, best_score = c, s
    return best  # type: ignore[return-value]


def pod_at_far(
    y_true: np.ndarray, p_pred: np.ndarray, *, far_max: float = 0.3,
) -> ConfusionStats | None:
    """Highest POD subject to FAR <= ``far_max`` (typical literature
    convention for HAB cost-sensitive reporting)."""
    grid = np.linspace(0.01, 0.99, 99)
    candidate = None
    for thr in grid:
        c = confusion_at(y_true, p_pred, thr)
        if np.isnan(c.far) or c.far > far_max:
            continue
        if candidate is None or c.pod > candidate.pod:
            candidate = c
    return candidate


# -----------------------------------------------------------------------
# One-shot report
# -----------------------------------------------------------------------
@dataclass
class EventReport:
    name: str
    n: int
    base_rate: float
    pr_auc: float
    brier: BrierDecomp
    best_csi: ConfusionStats
    best_f1: ConfusionStats
    pod_at_far_le_03: ConfusionStats | None
    reliability: pd.DataFrame = field(repr=False)

    def as_table(self) -> pd.DataFrame:
        rows = [
            ("name",              self.name),
            ("n",                 self.n),
            ("base_rate",         f"{self.base_rate:.4f}"),
            ("PR-AUC",            f"{self.pr_auc:.4f}"),
            ("Brier",             f"{self.brier.brier:.4f}"),
            ("  reliability",     f"{self.brier.reliability:.4f}"),
            ("  resolution",      f"{self.brier.resolution:.4f}"),
            ("  uncertainty",     f"{self.brier.uncertainty:.4f}"),
            ("best CSI thr",      f"{self.best_csi.threshold:.2f}  CSI={self.best_csi.csi:.3f}  POD={self.best_csi.pod:.3f}  FAR={self.best_csi.far:.3f}"),
            ("best F1 thr",       f"{self.best_f1.threshold:.2f}  F1={self.best_f1.f1:.3f}"),
        ]
        if self.pod_at_far_le_03:
            c = self.pod_at_far_le_03
            rows.append(("POD @ FAR<=0.3", f"thr={c.threshold:.2f}  POD={c.pod:.3f}"))
        else:
            rows.append(("POD @ FAR<=0.3", "no threshold satisfies FAR<=0.3"))
        return pd.DataFrame(rows, columns=["metric", "value"]).set_index("metric")


def binary_event_report(
    *, y_true, p_pred, name: str,
    frozen_thresholds: "dict[str, float] | None" = None,
) -> EventReport:
    """One-shot event report.

    Threshold selection is the most common source of test-fold leakage in
    HAB benchmarks: ``best_csi`` / ``best_f1`` / ``pod_at_far_le_03``
    sweep thresholds against the very ``(y_true, p_pred)`` they then
    score, so reporting them on a held-out fold inflates the operating
    point. To enforce the train -> val -> test discipline the README
    promises, accept a ``frozen_thresholds`` dict::

        {"csi": 0.27, "f1": 0.31, "far_le_03": 0.42}

    When provided, no sweeps are performed on this fold; the report
    reuses the val-tuned thresholds. Missing keys still fall back to a
    sweep on the current fold (so val itself can be tuned by passing
    ``frozen_thresholds=None``).
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    mask = ~(np.isnan(y) | np.isnan(p))
    y, p = y[mask], p[mask]
    if len(y) == 0:
        raise ValueError("no non-null pairs to score")
    ft = frozen_thresholds or {}

    if "csi" in ft:
        best_csi = confusion_at(y, p, float(ft["csi"]))
    else:
        best_csi = best_threshold(y, p, by="csi")
    if "f1" in ft:
        best_f1 = confusion_at(y, p, float(ft["f1"]))
    else:
        best_f1 = best_threshold(y, p, by="f1")
    if "far_le_03" in ft:
        pod_far = confusion_at(y, p, float(ft["far_le_03"]))
    else:
        pod_far = pod_at_far(y, p, far_max=0.3)

    return EventReport(
        name=name,
        n=len(y),
        base_rate=float(y.mean()),
        pr_auc=pr_auc(y, p),
        brier=brier_decomposition(y, p),
        best_csi=best_csi,
        best_f1=best_f1,
        pod_at_far_le_03=pod_far,
        reliability=reliability_table(y, p),
    )


# -----------------------------------------------------------------------
# Block bootstrap for spatially / temporally correlated samples
# -----------------------------------------------------------------------
def bootstrap_metric(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    *,
    metric: "str | callable" = "pr_auc",
    n_boot: int = 1000,
    by: "np.ndarray | None" = None,
    seed: int = 42,
    ci: float = 0.95,
) -> dict:
    """Bootstrap a metric with optional cluster (block) resampling.

    HAB station-day samples are NOT iid: rows from the same pier in
    consecutive weeks are highly correlated, and the pooled per-row
    bootstrap therefore underestimates the variance of any aggregate
    metric. Pass ``by=stations`` (or any cluster id array of length
    ``len(y_true)``) to resample CLUSTERS with replacement instead --
    this is the standard fix for spatially/temporally correlated
    forecast verification (e.g. Necker et al. 2024 spatial CIs).

    Returns ``{"point": float, "lo": float, "hi": float,
    "n_boot_effective": int, "metric": name}``.

    ``metric`` may be a callable taking ``(y, p) -> float`` or one of
    the strings ``"pr_auc"`` / ``"brier"``. ``ci`` is the central
    confidence-interval width (default 95%, percentile method).
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    mask = ~(np.isnan(y) | np.isnan(p))
    y, p = y[mask], p[mask]
    if by is not None:
        by_arr = np.asarray(by)[mask]
    else:
        by_arr = None

    if isinstance(metric, str):
        if metric == "pr_auc":
            fn = pr_auc
        elif metric == "brier":
            fn = lambda yy, pp: brier_decomposition(yy, pp).brier  # noqa: E731
        else:
            raise ValueError(f"unknown metric: {metric}")
        metric_name = metric
    else:
        fn = metric
        metric_name = getattr(metric, "__name__", "callable")

    point = float(fn(y, p)) if y.size else float("nan")
    if y.size == 0 or n_boot <= 0:
        return dict(point=point, lo=float("nan"), hi=float("nan"),
                    n_boot_effective=0, metric=metric_name)

    rng = np.random.default_rng(seed)
    samples: list[float] = []

    if by_arr is not None:
        clusters, cluster_inv = np.unique(by_arr, return_inverse=True)
        cluster_idx_lists = [
            np.flatnonzero(cluster_inv == k) for k in range(len(clusters))
        ]
        n_clusters = len(clusters)
        for _ in range(n_boot):
            picks = rng.integers(0, n_clusters, size=n_clusters)
            idx = np.concatenate([cluster_idx_lists[k] for k in picks])
            yb, pb = y[idx], p[idx]
            try:
                v = float(fn(yb, pb))
            except Exception:
                v = float("nan")
            if not np.isnan(v):
                samples.append(v)
    else:
        n = y.size
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            yb, pb = y[idx], p[idx]
            try:
                v = float(fn(yb, pb))
            except Exception:
                v = float("nan")
            if not np.isnan(v):
                samples.append(v)

    if not samples:
        return dict(point=point, lo=float("nan"), hi=float("nan"),
                    n_boot_effective=0, metric=metric_name)

    arr = np.asarray(samples)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(arr, alpha))
    hi = float(np.quantile(arr, 1.0 - alpha))
    return dict(point=point, lo=lo, hi=hi,
                n_boot_effective=int(arr.size), metric=metric_name)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.05, size=2000)
    p = np.clip(0.05 + 0.6 * y + rng.normal(0, 0.15, size=y.size), 0, 1)
    rep = binary_event_report(y_true=y, p_pred=p, name="synthetic")
    print(rep.as_table().to_string())
    ci = bootstrap_metric(y, p, metric="pr_auc", n_boot=200)
    print(f"PR-AUC bootstrap: {ci['point']:.3f}  "
          f"95% CI [{ci['lo']:.3f}, {ci['hi']:.3f}]")
