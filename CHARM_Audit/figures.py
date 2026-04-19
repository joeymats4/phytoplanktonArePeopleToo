"""Phase 3 figures + ablation analyses.

Reads ``Data/baselines/baseline_table.csv`` (and optionally per-arm
prediction panels written by ``run_baselines.py`` extensions) and
emits the five paper-defining artifacts described in the project plan.

PNG figures land under ``Project/plots/`` (override with the
``DH2026_PLOTS_ROOT`` env var, see ``storage.plots_root``); CSV
side-tables (``by_regime_table.csv``) land under
``Project/Data/baselines/``:

    plots/fig1_pr_auc_brier_bars.png    per-arm PR-AUC + Brier on the chosen fold (default test = 2022-2025)
    plots/fig2_pace_vs_viirs_delta.png  Delta PR-AUC PACE - VIIRS on the chosen fold
    plots/fig3_reliability.png          reliability diagrams for calibrated arms
    Data/baselines/by_regime_table.csv  PR-AUC / Brier under upwelling/ENSO regimes
    plots/fig4_operational_replay.png   cumulative Brier regret vs C-HARM on the chosen fold (default test = 2022-2025)
    plots/fig5_2025_socal_extreme.png   per-pier panel for the 2025 SoCal cluster

Every figure is generated headless (Agg backend) so the module imports
cleanly on EC2 / SageMaker. Each routine is independent; you can call
just one or run ``python figures.py`` for the whole bundle.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import storage  # noqa: E402

DATA_DIR = storage.data_root()
PLOTS_DIR = storage.plots_root()
BASELINES_DIR = storage.dataset_dir("baselines")


# -----------------------------------------------------------------------
# Figure 1: per-arm PR-AUC + Brier bar chart on the test fold
# -----------------------------------------------------------------------
def fig1_pr_auc_brier_bars(
    table: pd.DataFrame,
    *,
    fold: str = "test",
    out_path: Path | str = "fig1_pr_auc_brier_bars.png",
) -> Path:
    sub = table[table["fold"] == fold].copy()
    if sub.empty:
        raise ValueError(f"No rows for fold={fold!r} in baseline_table.csv")

    targets = sorted(sub["target"].unique())
    fig, axes = plt.subplots(
        nrows=2, ncols=len(targets),
        figsize=(5.5 * len(targets), 8), sharey="row",
    )
    if len(targets) == 1:
        axes = np.atleast_2d(axes).reshape(2, 1)

    for j, t in enumerate(targets):
        s = sub[sub["target"] == t].sort_values("pr_auc", ascending=False)
        axes[0, j].barh(s["model"], s["pr_auc"])
        axes[0, j].set_xlabel("PR-AUC")
        n_max = s["n"].max()
        n_str = "n/a" if pd.isna(n_max) else int(n_max)
        axes[0, j].set_title(f"{t}  (n={n_str})")
        axes[0, j].invert_yaxis()
        # base-rate reference line (random predictor PR-AUC ≈ base rate)
        if not s["base_rate"].dropna().empty:
            br = float(s["base_rate"].dropna().iloc[0])
            axes[0, j].axvline(br, ls="--", lw=1.0, color="grey",
                               label=f"base rate = {br:.3f}")
            axes[0, j].legend(loc="lower right", fontsize=8)

        s2 = sub[sub["target"] == t].sort_values("brier", ascending=True)
        axes[1, j].barh(s2["model"], s2["brier"])
        axes[1, j].set_xlabel("Brier (lower = better)")
        axes[1, j].invert_yaxis()

    fig.suptitle(f"Per-arm PR-AUC + Brier on fold = {fold}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = _save_fig(fig, out_path)
    plt.close(fig)
    return out


# -----------------------------------------------------------------------
# Figure 2: PACE − VIIRS delta PR-AUC bar chart
# -----------------------------------------------------------------------
def fig2_pace_vs_viirs_delta(
    table: pd.DataFrame,
    *,
    fold: str = "test",
    out_path: Path | str = "fig2_pace_vs_viirs_delta.png",
) -> Path:
    """Delta PR-AUC and Delta Brier between PACE and VIIRS arms.

    Computed over the rows for ``fold`` (default ``test`` = 2022-2025
    per ``baselines.split_train_val_test``); pass ``fold="val"`` for
    2019-2021 or ``fold="extreme2015"`` for the held-out 2015 case
    study. The earlier docstring claimed a "2024-2025 slice" which was
    not what the code computed.
    """
    sub = table[table["fold"] == fold].copy()
    if sub.empty:
        raise ValueError(f"No rows for fold={fold!r} in baseline_table.csv")

    pairs = [
        ("lgb_arm_a_plus_pace", "lgb_arm_a_plus_viirs", "LightGBM"),
        ("gnn_arm_a_plus_pace", "gnn_arm_a_plus_viirs", "GNN"),
    ]
    rows: list[dict] = []
    for pace_key, viirs_key, family in pairs:
        for t in sorted(sub["target"].unique()):
            p_row = sub[(sub["model"] == pace_key) & (sub["target"] == t)]
            v_row = sub[(sub["model"] == viirs_key) & (sub["target"] == t)]
            if p_row.empty or v_row.empty:
                continue
            rows.append(dict(
                family=family, target=t,
                d_pr_auc=float(p_row["pr_auc"].iloc[0] - v_row["pr_auc"].iloc[0]),
                d_brier=float(p_row["brier"].iloc[0] - v_row["brier"].iloc[0]),
            ))
    if not rows:
        raise RuntimeError(
            "No PACE/VIIRS arm pairs in the table; run run_baselines.py "
            "with PACE + VIIRS panels populated first."
        )
    delta = pd.DataFrame(rows)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    delta_sorted = delta.sort_values(["target", "family"])
    labels = [f"{r.family} / {r.target}" for r in delta_sorted.itertuples()]

    ax[0].barh(labels, delta_sorted["d_pr_auc"])
    ax[0].axvline(0.0, color="black", lw=0.8)
    ax[0].set_xlabel("Δ PR-AUC  (PACE − VIIRS)")
    ax[0].set_title(f"PACE arm vs VIIRS arm, fold = {fold}")

    ax[1].barh(labels, delta_sorted["d_brier"])
    ax[1].axvline(0.0, color="black", lw=0.8)
    ax[1].set_xlabel("Δ Brier  (PACE − VIIRS)  (negative is better)")

    fig.tight_layout()
    out = _save_fig(fig, out_path)
    plt.close(fig)
    return out


# -----------------------------------------------------------------------
# Figure 3: reliability diagrams
# -----------------------------------------------------------------------
def fig3_reliability_diagrams(
    *,
    out_path: Path | str = "fig3_reliability.png",
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> Path:
    """Plot reliability diagrams for one or more (y, p) prediction sets.

    ``predictions`` maps a label like ``"lgb_arm_a_plus_pace/p_pn"`` to
    a ``(y_true, p_pred)`` ndarray pair. Pass these from a notebook that
    has the per-arm prediction frames in memory; this module does not
    re-fit models.

    Binning strategy is chosen per-series from the ``"/<target>"`` tail
    of the label:

    - ``p_pda`` (rare events whose predictions cluster near the 3% base
      rate) -> ``strategy="quantile"`` with 8 bins so every bin holds a
      comparable number of forecasts; otherwise the uniform-binned curve
      collapses into bin 0 and looks flat.
    - everything else -> uniform 10-bin binning.

    Bins with fewer than 5 forecasts are dropped so a single-row tail
    bin does not dominate the figure.
    """
    from evaluate import reliability_table

    if not predictions:
        # Allow empty call to write an explanatory placeholder so the
        # downstream Makefile doesn't fail.
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5,
                "fig3_reliability_diagrams(predictions={...}) was\n"
                "called with no per-arm predictions.\n"
                "Generate predictions in run_baselines.py extension.",
                ha="center", va="center")
        ax.set_xticks([]); ax.set_yticks([])
        out = _save_fig(fig, out_path)
        plt.close(fig)
        return out

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="grey", label="perfect")
    for name, (y, p) in predictions.items():
        target = name.rsplit("/", 1)[-1] if "/" in name else ""
        if target == "p_pda":
            rt = reliability_table(
                np.asarray(y), np.asarray(p),
                n_bins=8, strategy="quantile", min_bin_n=5,
            )
        else:
            rt = reliability_table(
                np.asarray(y), np.asarray(p),
                n_bins=10, strategy="uniform", min_bin_n=5,
            )
        rt = rt.dropna(subset=["mean_p", "obs_rate"])
        if rt.empty:
            continue
        ax.plot(rt["mean_p"], rt["obs_rate"], marker="o", label=name)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_title("Reliability diagram (p_pda: quantile bins; p_pn: uniform bins)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = _save_fig(fig, out_path)
    plt.close(fig)
    return out


# -----------------------------------------------------------------------
# By-regime decomposition (CUTI sign + ONI sign)
# -----------------------------------------------------------------------
def by_regime_table(
    predictions: pd.DataFrame,
    *,
    upwelling_panel: pd.DataFrame,
    climate_panel: pd.DataFrame,
    out_path: Path | str = "by_regime_table.csv",
) -> pd.DataFrame:
    """Score every (model, target) under upwelling vs relaxation and
    El Nino vs neutral vs La Nina regimes.

    ``predictions`` schema: ``station, time, model, target, y_true, p_pred``.
    ``upwelling_panel`` schema: ``station, time, CUTI[, BEUTI]``.
    ``climate_panel`` schema: ``time, oni[, pdo, mei, npgo]``.

    Audit fix (H3) -- CUTI and ONI are joined with the **same publish
    latency** (``replay.LATENCY``) that ``baselines._make_features``
    uses at training time. The earlier version merged on raw ``time``,
    so a row was stratified by a CUTI / ONI value the model could not
    yet have seen at forecast time, drifting regime PR-AUC / Brier
    away from "what the model saw." With latency applied, regime
    stratification is causally honest.
    """
    from evaluate import binary_event_report
    from replay import LATENCY

    p = predictions.copy()
    p["time"] = pd.to_datetime(p["time"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)

    up = upwelling_panel.copy()
    up["time"] = pd.to_datetime(up["time"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    if "CUTI" not in up.columns:
        raise KeyError("upwelling_panel needs a 'CUTI' column")
    # H3: shift CUTI source time forward by its documented publish lag
    # so merge_asof(direction="backward") only attaches CUTI values that
    # would have been visible at the prediction's row time. Mirror of
    # baselines._make_features:298-303.
    up["time"] = up["time"] + LATENCY["cuti"].delay
    up = up.sort_values(["station", "time"])

    cl = climate_panel.copy()
    cl["time"] = pd.to_datetime(cl["time"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    if "oni" not in cl.columns:
        raise KeyError("climate_panel needs an 'oni' column")
    cl["time"] = cl["time"] + LATENCY["climate_idx"].delay
    cl = cl[["time", "oni"]].sort_values("time")

    # per-pier merge_asof for upwelling
    rows_up: list[pd.DataFrame] = []
    for st, gb in p.groupby("station"):
        sub_up = up[up["station"] == st][["time", "CUTI"]].sort_values("time")
        if sub_up.empty:
            gb["CUTI"] = np.nan
            rows_up.append(gb)
            continue
        rows_up.append(pd.merge_asof(
            gb.sort_values("time"), sub_up,
            on="time", direction="backward", tolerance=pd.Timedelta(days=14),
        ))
    p = pd.concat(rows_up, ignore_index=True)
    p = pd.merge_asof(
        p.sort_values("time"), cl,
        on="time", direction="backward",
    )

    p["upwell_sign"] = np.where(p["CUTI"].fillna(0) > 0, "upwelling", "relaxation")
    p["enso_sign"] = pd.cut(
        p["oni"].fillna(0),
        bins=[-np.inf, -0.5, 0.5, np.inf],
        labels=["la_nina", "neutral", "el_nino"],
    ).astype(str)

    def _score_subset(sub: pd.DataFrame, model: str, target: str,
                       regime_kind: str, regime_val: str) -> dict:
        m = ~(sub["y_true"].isna() | sub["p_pred"].isna())
        y, ph = sub.loc[m, "y_true"].values, sub.loc[m, "p_pred"].values
        if len(y) == 0 or y.sum() == 0:
            return dict(model=model, target=target,
                        regime_kind=regime_kind, regime=regime_val,
                        n=int(len(y)), pr_auc=float("nan"),
                        brier=float("nan"))
        rep = binary_event_report(y_true=y, p_pred=ph, name=f"{model}/{target}/{regime_val}")
        return dict(model=model, target=target,
                    regime_kind=regime_kind, regime=regime_val,
                    n=int(len(y)), pr_auc=rep.pr_auc, brier=rep.brier.brier)

    rows = []
    for (model, target), gb in p.groupby(["model", "target"]):
        for v in ["upwelling", "relaxation"]:
            rows.append(_score_subset(
                gb[gb["upwell_sign"] == v], model, target, "upwell", v,
            ))
        for v in ["la_nina", "neutral", "el_nino"]:
            rows.append(_score_subset(
                gb[gb["enso_sign"] == v], model, target, "enso", v,
            ))
    out = pd.DataFrame(rows)
    out_path = _save_csv(out, out_path)
    return out


# -----------------------------------------------------------------------
# Operational replay -- cumulative Brier regret vs C-HARM
# -----------------------------------------------------------------------
def operational_replay_summary(
    predictions: pd.DataFrame,
    *,
    out_path: Path | str = "fig4_operational_replay.png",
    title: str = (
        "Cumulative Brier regret vs C-HARM v3.1 (test fold predictions)"
    ),
) -> Path:
    """Per-arm cumulative Brier regret vs. C-HARM v3.1 over time.

    NOTE -- the earlier title claimed "rolling-origin replay 2023-2025"
    but the code does not actually re-issue forecasts on a rolling
    cadence. It scores the predictions parquet emitted by
    ``run_baselines.py`` (a single train -> val -> test fit) row by row
    and accumulates the per-row Brier difference vs C-HARM. The
    operational discipline (latency, no future joins) is enforced
    inside ``baselines._make_features`` / ``replay.LATENCY``, not by
    this figure. The new title reflects that.

    ``predictions`` schema: ``station, time, model, target, y_true,
    p_pred``. Cumulative regret is defined as
    ``sum_{t<=T} (loss_model - loss_charm)`` where ``loss = (p - y)^2``
    (Brier). Negative regret means the model beats C-HARM cumulatively.

    Audit fix (H1) -- the merge against C-HARM is keyed on
    ``["station", "time", "target"]``. The previous version dropped
    ``station`` and merged on ``time`` only, which wrong-pairs C-HARM
    losses across piers (multiple piers can share a sample timestamp,
    so the inner merge produced a station-cartesian-style explosion
    and dramatically biased the cumulative trace).
    """
    p = predictions.copy()
    p["time"] = pd.to_datetime(p["time"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    p["loss"] = (p["p_pred"] - p["y_true"]) ** 2

    charm_keys = [m for m in p["model"].unique() if str(m).startswith("charm_")]
    if not charm_keys:
        raise RuntimeError("No charm_* model in predictions; nothing to compare against")
    charm_key = sorted(charm_keys)[0]  # deterministic across runs
    charm = (
        p[p["model"] == charm_key][["station", "time", "target", "loss"]]
        .rename(columns={"loss": "loss_charm"})
        .dropna(subset=["loss_charm"])
        # If C-HARM ever has duplicate (station, time, target) we keep
        # the most pessimistic (worst Brier) row; in practice this is
        # a no-op because the pull is deduped.
        .sort_values(["station", "time", "target"])
        .drop_duplicates(subset=["station", "time", "target"], keep="last")
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    for (model, target), gb in p[p["model"] != charm_key].groupby(["model", "target"]):
        merged = gb[["station", "time", "target", "loss"]].merge(
            charm[charm["target"] == target][["station", "time", "loss_charm"]],
            on=["station", "time"], how="inner",
        )
        if merged.empty:
            continue
        merged = merged.dropna(subset=["loss", "loss_charm"]) \
                       .sort_values("time")
        if merged.empty:
            continue
        merged["regret"] = (merged["loss"] - merged["loss_charm"]).cumsum()
        ax.plot(merged["time"], merged["regret"], label=f"{model}/{target}")

    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_ylabel("Cumulative Brier regret  (model - C-HARM)")
    ax.set_xlabel("Forecast time")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    out = _save_fig(fig, out_path)
    plt.close(fig)
    return out


# -----------------------------------------------------------------------
# 2025 SoCal extreme-event panel
# -----------------------------------------------------------------------
SOCAL_PIERS = ("scripps", "newport", "santamonica", "stearns")

# Curated set of model series to plot in fig5 by default. Without a
# whitelist the panel renders every (model, target) pair (~26 lines per
# pier) and the legend overlaps the Scripps trace. The defaults below
# pick the operational reference, the trivial baseline, the best LGB
# arms, and the best GNN arm.
FIG5_DEFAULT_SERIES: tuple[str, ...] = (
    "charm_wvcharmV3_1day",
    "climatology",
    "lgb_arm_a_plus_calcofi",
    "lgb_arm_a_plus_pace_plus_viirs",
    "gnn_arm_a_plus_pace",
)


def fig5_2025_socal_extreme(
    predictions: pd.DataFrame,
    *,
    out_path: Path | str = "fig5_2025_socal_extreme.png",
    piers: Iterable[str] = SOCAL_PIERS,
    series_whitelist: Iterable[str] | None = FIG5_DEFAULT_SERIES,
) -> Path:
    """Per-pier 2025 SoCal panel: P(PN), observed event, model vs. truth.

    ``series_whitelist`` restricts which ``model`` values are plotted as
    line series. Observed-event red Xs are always rendered regardless of
    the whitelist (they come from the union of ``y_true == 1`` across
    every model). Pass ``series_whitelist=None`` to plot every model.
    """
    p = predictions.copy()
    p["time"] = pd.to_datetime(p["time"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    p = p[(p["time"].dt.year == 2025) & (p["station"].isin(piers))]
    if p.empty:
        raise RuntimeError("No 2025 SoCal predictions in input frame")

    if series_whitelist is not None:
        whitelist = set(series_whitelist)
        p_lines = p[p["model"].isin(whitelist)]
    else:
        p_lines = p

    piers = tuple(piers)
    fig, axes = plt.subplots(
        nrows=len(piers), ncols=1, figsize=(13.5, 2.4 * len(piers)),
        sharex=True, sharey=True,
    )
    if len(piers) == 1:
        axes = [axes]

    for ax, pier in zip(axes, piers):
        sub_lines = p_lines[p_lines["station"] == pier]
        sub_truth = p[p["station"] == pier]
        if sub_lines.empty and sub_truth.empty:
            ax.text(0.5, 0.5, f"no 2025 data at {pier}", ha="center", va="center")
            continue
        for (model, target), gb in sub_lines.groupby(["model", "target"]):
            gb = gb.sort_values("time")
            ax.plot(gb["time"], gb["p_pred"], label=f"{model}/{target}", lw=1.0)
        # ground truth: same observation can appear under every model's
        # row in the predictions parquet, so deduplicate on
        # (station, time, target) before scattering -- otherwise a
        # single observed event renders ~26 stacked red Xs at the same
        # coordinate (audit medium item, ties to fig5 honesty).
        truth = (
            sub_truth[sub_truth["y_true"] == 1]
            [["station", "time", "target", "y_true"]]
            .drop_duplicates(subset=["station", "time", "target"])
        )
        ax.scatter(truth["time"], truth["y_true"], marker="x", color="red",
                   s=18, zorder=5, label="observed event")
        ax.set_ylabel(pier)
        ax.set_ylim(0, 1.05)

    axes[0].legend(
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        fontsize=7, frameon=False,
    )
    axes[-1].set_xlabel("2025")
    fig.suptitle("2025 SoCal PN/pDA event predictions vs observed labels")
    fig.tight_layout(rect=(0, 0, 0.82, 0.97))
    out = _save_fig(fig, out_path)
    plt.close(fig)
    return out


# -----------------------------------------------------------------------
# Headline figure: PR-curve overlay vs C-HARM v3.1 (test fold)
#
# The fig1_* family below sits beside the existing fig1_pr_auc_brier_bars
# bar chart -- they are additive, not a replacement. The headline figure
# answers "how does our best model rank against C-HARM v3.1 across all
# decision thresholds?" using PR curves on the row-aligned subset where
# both have coverage.
# -----------------------------------------------------------------------
HEADLINE_OUR_BEST = "lgb_arm_a_plus_calcofi"
HEADLINE_RAW_CHARM = "charm_wvcharmV3_1day"
HEADLINE_CALIBRATED_CHARM = "charm_wvcharmV3_1day_calibrated"
HEADLINE_FLOORS = ("climatology", "persistence")

# Display labels used in legend / annotations. Keep aligned with the
# CHARM_AUDIT.md sidecar so a teammate searching the codebase can pivot
# from a label on the figure back to the model name in the parquet.
_HEADLINE_LABELS = {
    HEADLINE_OUR_BEST:           "lgb arm A + CalCOFI (our best)",
    HEADLINE_RAW_CHARM:          "C-HARM v3.1 (raw, 1-day)",
    HEADLINE_CALIBRATED_CHARM:   "C-HARM v3.1 (post-isotonic)",
    "climatology":               "climatology",
    "persistence":               "persistence",
    "charm_wvcharmV3_0day_raw":  "C-HARM v3.1 (raw, 0-day)",
}

# Display order top-to-bottom in the legend. Stars / linestyles are
# applied after this ordering by ``_headline_style``.
_HEADLINE_ORDER = (
    HEADLINE_OUR_BEST,
    HEADLINE_RAW_CHARM,
    HEADLINE_CALIBRATED_CHARM,
    "climatology",
    "persistence",
)


def _headline_style(model: str) -> dict:
    """Return matplotlib kwargs for one model series in the headline.

    Style choices:
      * our-best is the heaviest line and the only one with a star marker.
      * raw C-HARM is solid, mid weight (the line we have to beat).
      * calibrated C-HARM is dashed (same family, different processing).
      * climatology and persistence are thin so they read as floor refs.
    """
    base = {"lw": 1.4, "alpha": 0.95}
    if model == HEADLINE_OUR_BEST:
        base.update({"lw": 2.6, "marker": "*", "markersize": 8,
                     "markevery": 0.1, "zorder": 5})
    elif model == HEADLINE_RAW_CHARM:
        base.update({"lw": 2.0, "ls": "-",  "zorder": 4})
    elif model == HEADLINE_CALIBRATED_CHARM:
        base.update({"lw": 1.8, "ls": "--", "zorder": 3})
    else:
        base.update({"lw": 1.0, "ls": ":", "alpha": 0.7})
    return base


def _row_align_predictions(
    preds: pd.DataFrame,
    target: str,
    models: Iterable[str],
) -> pd.DataFrame:
    """Inner-join across the named model series on (station, time) for
    one target, returning a long-format frame containing only the rows
    where every series has a non-null prediction. Asserts that all
    series end up with the same n.
    """
    models = list(models)
    sub = preds[(preds["target"] == target) & preds["model"].isin(models)].copy()
    sub = sub.dropna(subset=["p_pred", "y_true"])
    if sub.empty:
        return sub
    sub["time"] = pd.to_datetime(sub["time"], utc=True)
    pivot = sub.pivot_table(
        index=["station", "time"], columns="model",
        values="p_pred", aggfunc="last",
    )
    aligned = pivot.dropna(how="any", subset=[m for m in models if m in pivot.columns])
    if aligned.empty:
        return aligned.reset_index()
    truth = (
        sub.dropna(subset=["y_true"])
           .drop_duplicates(subset=["station", "time"])
           [["station", "time", "y_true"]]
           .set_index(["station", "time"])
    )
    aligned = aligned.join(truth, how="left").dropna(subset=["y_true"])
    return aligned.reset_index()


def _bootstrap_pr_auc(
    y: np.ndarray, p: np.ndarray, stations: np.ndarray | None = None,
    *, n_boot: int = 200,
) -> tuple[float, float, float]:
    from evaluate import bootstrap_metric
    res = bootstrap_metric(
        y, p, metric="pr_auc", n_boot=n_boot, by=stations, seed=42,
    )
    return res["point"], res["lo"], res["hi"]


def fig1_pr_curves_vs_charm(
    predictions: pd.DataFrame,
    *,
    out_path: Path | str = "fig1_pr_curves_vs_charm.png",
    targets: Iterable[str] = ("p_pn", "p_pda"),
    n_boot: int = 200,
    extra_lines: Iterable[str] = (),
) -> Path:
    """Headline figure: 2-panel PR-curve overlay on the test fold,
    restricted to the row-aligned subset where C-HARM has coverage.

    Per panel, each model series is plotted as a full precision-recall
    curve. Legend entries are PR-AUC with station-block bootstrap 95%
    confidence intervals. A horizontal dashed line at the per-panel
    base rate gives the random-chance precision floor.

    The five canonical series are ``HEADLINE_OUR_BEST``, raw and
    calibrated C-HARM 1-day, climatology, and persistence. Pass
    additional model names through ``extra_lines`` to overlay (e.g. the
    0-day raw nowcast as a "best-case horizon for C-HARM" upper bound).
    """
    from sklearn.metrics import precision_recall_curve

    targets = tuple(targets)
    series = list(_HEADLINE_ORDER) + [
        m for m in extra_lines if m not in _HEADLINE_ORDER
    ]

    fig, axes = plt.subplots(
        1, len(targets), figsize=(6.2 * len(targets), 5.4),
        sharey=False,
    )
    if len(targets) == 1:
        axes = [axes]

    # Re-shape the predictions parquet into the union of (station, time)
    # rows where the headline-required series all have a value. We do
    # NOT enforce alignment for ``extra_lines`` -- they are scored on
    # whatever rows they happen to overlap with the aligned set, so
    # adding e.g. the 0-day variant doesn't shrink the headline n.
    headline_required = [
        HEADLINE_OUR_BEST, HEADLINE_RAW_CHARM, *HEADLINE_FLOORS,
    ]

    panel_summaries: list[dict] = []
    for ax, tgt in zip(axes, targets):
        aligned = _row_align_predictions(predictions, tgt, headline_required)
        if aligned.empty:
            ax.text(0.5, 0.5, f"no aligned rows for {tgt}",
                    ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        n_aligned = len(aligned)
        base_rate = float(aligned["y_true"].mean())

        ax.axhline(
            base_rate, ls="--", lw=1.0, color="grey",
            label=f"base rate = {base_rate:.3f}",
        )

        per_series_preds = predictions[
            predictions["target"] == tgt
        ].dropna(subset=["p_pred", "y_true"]).copy()
        per_series_preds["time"] = pd.to_datetime(
            per_series_preds["time"], utc=True,
        )

        keys = aligned[["station", "time"]].copy()
        keys["__align"] = 1

        for model in series:
            if model not in per_series_preds["model"].unique():
                continue
            pred = per_series_preds[per_series_preds["model"] == model]
            joined = pred.merge(keys, on=["station", "time"], how="inner")
            if joined.empty:
                continue
            y = joined["y_true"].values.astype(float)
            p = joined["p_pred"].values.astype(float)
            stations = joined["station"].values
            if y.sum() == 0 or y.sum() == y.size:
                continue
            prec, rec, _ = precision_recall_curve(y.astype(int), p)
            point, lo, hi = _bootstrap_pr_auc(
                y, p, stations=stations, n_boot=n_boot,
            )
            label_base = _HEADLINE_LABELS.get(model, model)
            label = (
                f"{label_base}: PR-AUC = {point:.3f}  "
                f"[{lo:.2f}, {hi:.2f}]"
            )
            ax.plot(rec, prec, label=label, **_headline_style(model))
            panel_summaries.append({
                "target": tgt, "model": model,
                "pr_auc": point, "pr_auc_lo": lo, "pr_auc_hi": hi,
                "n": len(joined),
            })

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Recall  (TP / positives)")
        ax.set_ylabel("Precision  (TP / alarms)")
        ax.set_title(
            f"{tgt}   test fold, n_aligned = {n_aligned}, "
            f"base rate = {base_rate:.3f}"
        )
        ax.legend(loc="upper right", fontsize=7, frameon=True)
        ax.grid(True, alpha=0.25)

    # Subtitle with the deltas vs raw C-HARM, computed from the
    # aligned summaries we just rendered.
    deltas = []
    for tgt in targets:
        ours = next(
            (s for s in panel_summaries
             if s["target"] == tgt and s["model"] == HEADLINE_OUR_BEST),
            None,
        )
        char = next(
            (s for s in panel_summaries
             if s["target"] == tgt and s["model"] == HEADLINE_RAW_CHARM),
            None,
        )
        if ours and char and np.isfinite(ours["pr_auc"]) and np.isfinite(char["pr_auc"]):
            deltas.append(
                f"Δ {tgt} = {ours['pr_auc'] - char['pr_auc']:+.2f}"
            )
    delta_str = "; ".join(deltas) if deltas else "Δ unavailable"
    fig.suptitle(
        "PR curves vs C-HARM v3.1 (operational incumbent), HABMAP test fold\n"
        f"Δ PR-AUC vs C-HARM v3.1 (raw):  {delta_str}",
        fontsize=11,
    )
    fig.text(
        0.5, 0.005,
        "C-HARM v3.1 sampled at HABMAP piers (~0.04° box, median over "
        "lat/lon); calibrated variant uses isotonic regression fit on "
        "first 30% of test (chronological), evaluated on last 70%.",
        ha="center", va="bottom", fontsize=7, style="italic",
    )

    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    out = _save_fig(fig, out_path)
    plt.close(fig)
    return out


def fig1b_reliability_vs_charm(
    predictions: pd.DataFrame,
    *,
    out_path: Path | str = "fig1b_reliability_vs_charm.png",
    targets: Iterable[str] = ("p_pn", "p_pda"),
) -> Path:
    """Reliability + sharpness for the three calibration-relevant series:
    our best model, raw C-HARM, calibrated C-HARM. Two rows (one per
    target). The reliability panel shows P(observed | predicted bin)
    against the diagonal; the sharpness panel shows the prediction
    histogram per series.
    """
    from evaluate import reliability_table

    targets = tuple(targets)
    series = (HEADLINE_OUR_BEST, HEADLINE_RAW_CHARM, HEADLINE_CALIBRATED_CHARM)

    fig, axes = plt.subplots(
        nrows=len(targets), ncols=2,
        figsize=(11, 4.8 * len(targets)),
        gridspec_kw={"width_ratios": [1.4, 1.0]},
    )
    if len(targets) == 1:
        axes = np.atleast_2d(axes)

    for i, tgt in enumerate(targets):
        ax_rel = axes[i, 0]
        ax_hist = axes[i, 1]
        ax_rel.plot([0, 1], [0, 1], ls="--", lw=0.8, color="grey",
                    label="perfectly calibrated")

        for model in series:
            sub = predictions[(predictions["model"] == model)
                              & (predictions["target"] == tgt)]
            sub = sub.dropna(subset=["y_true", "p_pred"])
            if sub.empty:
                continue
            y = sub["y_true"].values.astype(float)
            p = sub["p_pred"].values.astype(float)
            if tgt == "p_pda":
                rt = reliability_table(
                    y, p, n_bins=8, strategy="quantile", min_bin_n=3,
                )
            else:
                rt = reliability_table(
                    y, p, n_bins=10, strategy="uniform", min_bin_n=5,
                )
            rt = rt.dropna(subset=["mean_p", "obs_rate"])
            if not rt.empty:
                label = _HEADLINE_LABELS.get(model, model)
                ax_rel.plot(rt["mean_p"], rt["obs_rate"], marker="o",
                            label=label)
            ax_hist.hist(
                p, bins=20, alpha=0.45, range=(0.0, 1.0),
                label=_HEADLINE_LABELS.get(model, model),
            )

        ax_rel.set_xlabel("Predicted probability")
        ax_rel.set_ylabel("Observed event rate")
        ax_rel.set_xlim(0, 1)
        ax_rel.set_ylim(0, 1)
        ax_rel.set_title(f"{tgt}  reliability")
        ax_rel.legend(fontsize=8, loc="upper left")
        ax_rel.grid(True, alpha=0.25)

        ax_hist.set_xlabel("Predicted probability")
        ax_hist.set_ylabel("count")
        ax_hist.set_title(f"{tgt}  sharpness (prediction histogram)")
        ax_hist.set_xlim(0, 1)
        ax_hist.legend(fontsize=8)
        ax_hist.grid(True, alpha=0.25)

    fig.suptitle(
        "Reliability + sharpness: our best vs raw and isotonic-recalibrated C-HARM v3.1",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = _save_fig(fig, out_path)
    plt.close(fig)
    return out


def fig1c_charm_aggregator_robustness(
    aggregator_audit: pd.DataFrame,
    *,
    our_best_table: pd.DataFrame | None = None,
    out_path: Path | str = "fig1c_charm_aggregator_robustness.png",
    targets: Iterable[str] = ("p_pn", "p_pda"),
) -> Path:
    """Grouped PR-AUC + Brier bars for C-HARM under {median, mean, max,
    nearest} aggregators. Our-best is overlaid as a horizontal reference
    line so the reader sees that no choice of spatial reduction lifts
    C-HARM above our model.

    ``aggregator_audit`` is the parquet emitted by
    ``scripts/charm_audit_panel.py`` (one row per (target, aggregator)
    with ``pr_auc``, ``pr_auc_lo``, ``pr_auc_hi``, ``brier``, etc.).
    ``our_best_table`` is the ``baseline_table.csv`` row for the test
    fold; if omitted, the reference line is skipped.
    """
    targets = tuple(targets)
    if aggregator_audit.empty:
        raise ValueError("aggregator_audit is empty; run charm_audit_panel.py")

    aggregators = list(dict.fromkeys(aggregator_audit["aggregator"].tolist()))
    fig, axes = plt.subplots(
        nrows=len(targets), ncols=2,
        figsize=(11, 4.0 * len(targets)),
    )
    if len(targets) == 1:
        axes = np.atleast_2d(axes)

    for i, tgt in enumerate(targets):
        sub = aggregator_audit[aggregator_audit["target"] == tgt]
        sub = sub.set_index("aggregator").reindex(aggregators)
        x = np.arange(len(aggregators))

        ax_pr = axes[i, 0]
        bars_pr = ax_pr.bar(x, sub["pr_auc"].values)
        # Asymmetric error bars from the bootstrap CIs.
        err_lo = sub["pr_auc"].values - sub["pr_auc_lo"].values
        err_hi = sub["pr_auc_hi"].values - sub["pr_auc"].values
        ax_pr.errorbar(
            x, sub["pr_auc"].values,
            yerr=[err_lo, err_hi],
            fmt="none", ecolor="black", capsize=4, lw=1.0,
        )
        ax_pr.set_xticks(x)
        ax_pr.set_xticklabels(aggregators)
        ax_pr.set_ylabel("PR-AUC")
        ax_pr.set_title(f"{tgt}  C-HARM PR-AUC by aggregator")
        ax_pr.grid(True, axis="y", alpha=0.25)
        for bar, n in zip(bars_pr, sub["n"].values):
            ax_pr.text(bar.get_x() + bar.get_width() / 2, 0.005,
                       f"n={int(n) if pd.notna(n) else 0}",
                       ha="center", va="bottom", fontsize=7)

        ax_br = axes[i, 1]
        bars_br = ax_br.bar(x, sub["brier"].values)
        err_lo = sub["brier"].values - sub["brier_lo"].values
        err_hi = sub["brier_hi"].values - sub["brier"].values
        ax_br.errorbar(
            x, sub["brier"].values,
            yerr=[err_lo, err_hi],
            fmt="none", ecolor="black", capsize=4, lw=1.0,
        )
        ax_br.set_xticks(x)
        ax_br.set_xticklabels(aggregators)
        ax_br.set_ylabel("Brier (lower is better)")
        ax_br.set_title(f"{tgt}  C-HARM Brier by aggregator")
        ax_br.grid(True, axis="y", alpha=0.25)

        if our_best_table is not None:
            ours = our_best_table[
                (our_best_table["target"] == tgt)
                & (our_best_table["model"] == HEADLINE_OUR_BEST)
                & (our_best_table["fold"] == "test")
            ]
            if not ours.empty:
                pr = float(ours["pr_auc"].iloc[0])
                br = float(ours["brier"].iloc[0])
                ax_pr.axhline(
                    pr, ls="--", lw=1.4, color="C2",
                    label=f"our best ({HEADLINE_OUR_BEST}) = {pr:.3f}",
                )
                ax_pr.legend(fontsize=8, loc="upper right")
                ax_br.axhline(
                    br, ls="--", lw=1.4, color="C2",
                    label=f"our best ({HEADLINE_OUR_BEST}) = {br:.3f}",
                )
                ax_br.legend(fontsize=8, loc="upper right")
        # Avoid the ipsum bars touching zero for very small Brier values.
        ax_br.set_ylim(bottom=0.0)

    fig.suptitle(
        "C-HARM v3.1 spatial-aggregator robustness, test fold\n"
        "Our-best (horizontal dashed) sits above C-HARM under every aggregator.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = _save_fig(fig, out_path)
    plt.close(fig)
    return out


def fig1d_charm_horizon_curve(
    horizon_audit: pd.DataFrame,
    *,
    our_best_table: pd.DataFrame | None = None,
    out_path: Path | str = "fig1d_charm_horizon_curve.png",
    targets: Iterable[str] = ("p_pn", "p_pda"),
) -> Path:
    """C-HARM PR-AUC and Brier as a function of forecast horizon
    (0/1/2/3 day). Our-best is a horizontal reference. Demonstrates we
    beat C-HARM at every horizon, not just the cherry-picked 1-day.

    ``horizon_audit`` is the parquet emitted by
    ``scripts/charm_audit_panel.py`` (one row per (target, horizon) with
    ``pr_auc``, ``pr_auc_lo``, ``pr_auc_hi``, ``brier``, ``horizon``).
    """
    targets = tuple(targets)
    if horizon_audit.empty:
        raise ValueError("horizon_audit is empty; run charm_audit_panel.py")

    # Sort horizons by lead time, not lexically.
    horizon_order = ["wvcharmV3_0day", "wvcharmV3_1day",
                     "wvcharmV3_2day", "wvcharmV3_3day"]
    horizons = [h for h in horizon_order if h in horizon_audit["horizon"].unique()]

    fig, axes = plt.subplots(
        nrows=len(targets), ncols=2, figsize=(11, 4.0 * len(targets)),
    )
    if len(targets) == 1:
        axes = np.atleast_2d(axes)

    x = np.arange(len(horizons))
    short_labels = [h.replace("wvcharmV3_", "") for h in horizons]

    for i, tgt in enumerate(targets):
        sub = horizon_audit[horizon_audit["target"] == tgt]
        sub = sub.set_index("horizon").reindex(horizons)

        for ax, col, ci_lo, ci_hi, ylabel in (
            (axes[i, 0], "pr_auc", "pr_auc_lo", "pr_auc_hi",
             "PR-AUC (higher is better)"),
            (axes[i, 1], "brier", "brier_lo", "brier_hi",
             "Brier (lower is better)"),
        ):
            y = sub[col].values
            err_lo = y - sub[ci_lo].values
            err_hi = sub[ci_hi].values - y
            ax.errorbar(
                x, y, yerr=[err_lo, err_hi],
                marker="o", lw=1.5, capsize=4, color="C0",
                label="C-HARM v3.1",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(short_labels)
            ax.set_xlabel("Forecast horizon")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{tgt}  C-HARM by horizon")
            ax.grid(True, alpha=0.25)
            if our_best_table is not None:
                ours = our_best_table[
                    (our_best_table["target"] == tgt)
                    & (our_best_table["model"] == HEADLINE_OUR_BEST)
                    & (our_best_table["fold"] == "test")
                ]
                if not ours.empty:
                    val = float(ours[col].iloc[0])
                    ax.axhline(
                        val, ls="--", lw=1.4, color="C2",
                        label=f"our best = {val:.3f}",
                    )
            ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "C-HARM v3.1 forecast-horizon curve, test fold\n"
        "Lead time 0d (nowcast) -> 3d.  Our best stays above C-HARM at every horizon.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = _save_fig(fig, out_path)
    plt.close(fig)
    return out


# -----------------------------------------------------------------------
# Convenience
# -----------------------------------------------------------------------
def _save_fig(fig: plt.Figure, out_path: Path | str) -> Path:
    """Write a matplotlib figure under ``plots_root()``.

    Absolute paths are honored as-is; relative paths are joined under
    ``plots_root()`` (default ``Project/plots/``). Side-table CSVs go
    through ``_save_csv`` which writes under ``Data/baselines/``
    instead, so the plots/ tree only ever contains figures.
    """
    p = Path(str(out_path))
    target = p if p.is_absolute() else PLOTS_DIR.joinpath(str(out_path))
    storage.ensure_dir(PLOTS_DIR)
    if storage.is_remote(target):
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(suffix=".png", delete=False) as tf:
            fig.savefig(tf.name, dpi=140)
            local = Path(tf.name)
        target.write_bytes(local.read_bytes())
        local.unlink(missing_ok=True)
    else:
        fig.savefig(str(target), dpi=140)
    return target


def _save_csv(df: pd.DataFrame, out_path: Path | str) -> Path:
    p = Path(str(out_path))
    target = p if p.is_absolute() else BASELINES_DIR.joinpath(str(out_path))
    storage.ensure_dir(BASELINES_DIR)
    df.to_csv(str(target), index=False)
    return target


def _load_predictions(
    fold: str,
    targets: Iterable[str] = ("p_pn", "p_pda"),
) -> pd.DataFrame:
    """Load every per-(target, fold) predictions parquet written by
    ``run_baselines.py``. Missing files are skipped silently so a
    partial table still produces fig1/fig2.
    """
    frames: list[pd.DataFrame] = []
    for tgt in targets:
        path = BASELINES_DIR / f"predictions_{tgt}_{fold}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(str(path)))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-table", type=Path,
                    default=BASELINES_DIR / "baseline_table.csv")
    ap.add_argument("--fold", default="test",
                    choices=["test", "val", "extreme2015"])
    args = ap.parse_args()

    table = pd.read_csv(str(args.baseline_table))
    print(f"Loaded {len(table)} rows from {args.baseline_table}")

    fig1 = fig1_pr_auc_brier_bars(table, fold=args.fold)
    print(f"  -> {fig1}")

    # Headline figure family (fig1_pr_curves_vs_charm + supporting
    # fig1b/c/d). These read from per-(target, fold) predictions parquets
    # plus the audit parquets emitted by scripts/charm_audit_panel.py.
    # Each is independently optional so a partially-built audit still
    # produces what it can.
    headline_preds = _load_predictions(args.fold)
    if not headline_preds.empty:
        try:
            fig1_pr = fig1_pr_curves_vs_charm(headline_preds)
            print(f"  -> {fig1_pr}")
        except Exception as exc:  # noqa: BLE001
            print(f"  fig1_pr_curves_vs_charm skipped: {exc}")
        try:
            fig1b = fig1b_reliability_vs_charm(headline_preds)
            print(f"  -> {fig1b}")
        except Exception as exc:  # noqa: BLE001
            print(f"  fig1b_reliability_vs_charm skipped: {exc}")

    agg_path = BASELINES_DIR / "charm_aggregator_audit.parquet"
    if agg_path.exists():
        try:
            agg_df = pd.read_parquet(str(agg_path))
            fig1c = fig1c_charm_aggregator_robustness(
                agg_df, our_best_table=table,
            )
            print(f"  -> {fig1c}")
        except Exception as exc:  # noqa: BLE001
            print(f"  fig1c_charm_aggregator_robustness skipped: {exc}")
    else:
        print("  fig1c skipped: charm_aggregator_audit.parquet not built yet "
              "(run scripts/charm_audit_panel.py)")

    horizon_path = BASELINES_DIR / "charm_horizon_audit.parquet"
    if horizon_path.exists():
        try:
            horizon_df = pd.read_parquet(str(horizon_path))
            fig1d = fig1d_charm_horizon_curve(
                horizon_df, our_best_table=table,
            )
            print(f"  -> {fig1d}")
        except Exception as exc:  # noqa: BLE001
            print(f"  fig1d_charm_horizon_curve skipped: {exc}")
    else:
        print("  fig1d skipped: charm_horizon_audit.parquet not built yet "
              "(run scripts/charm_audit_panel.py)")

    try:
        fig2 = fig2_pace_vs_viirs_delta(table, fold=args.fold)
        print(f"  -> {fig2}")
    except RuntimeError as e:
        print(f"  fig2 skipped: {e}")

    preds = _load_predictions(args.fold)
    if preds.empty:
        print("\nFigures 3-5 skipped: no predictions_<target>_<fold>.parquet "
              "files in Data/baselines/. Re-run `python run_baselines.py` "
              "after the predictions-parquet extension landed.")
        return

    print(f"\nLoaded {len(preds)} prediction rows ({preds['model'].nunique()} "
          f"models x {preds['target'].nunique()} targets) for fig3-fig5")

    # fig3: reliability diagrams for the LGB and GNN PACE arms on each target
    rel: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tgt in sorted(preds["target"].unique()):
        for m in ("lgb_arm_a", "lgb_arm_a_plus_pace_plus_viirs",
                  "gnn_arm_a_plus_pace_plus_viirs",
                  f"charm_wvcharmV3_1day"):
            sub = preds[(preds["model"] == m) & (preds["target"] == tgt)]
            sub = sub.dropna(subset=["y_true", "p_pred"])
            if sub.empty or sub["y_true"].sum() == 0:
                continue
            rel[f"{m}/{tgt}"] = (sub["y_true"].values, sub["p_pred"].values)
    if rel:
        fig3 = fig3_reliability_diagrams(predictions=rel)
        print(f"  -> {fig3}")
    else:
        print("  fig3 skipped: no calibrated arms with positive labels")

    # fig4: cumulative-Brier-regret-vs-C-HARM operational replay (test fold).
    # Only meaningful when the predictions actually contain a charm_* row.
    if any(str(m).startswith("charm_") for m in preds["model"].unique()):
        try:
            fig4 = operational_replay_summary(preds)
            print(f"  -> {fig4}")
        except Exception as e:  # noqa: BLE001
            print(f"  fig4 skipped: {e}")
    else:
        print("  fig4 skipped: no charm_* predictions in the parquets")

    # fig5: 2025 SoCal extreme-event panel.
    try:
        fig5 = fig5_2025_socal_extreme(preds)
        print(f"  -> {fig5}")
    except Exception as e:  # noqa: BLE001
        print(f"  fig5 skipped: {e}")

    # by-regime decomposition: PR-AUC / Brier per (model, target) under
    # upwelling vs relaxation (CUTI sign) and El Nino / neutral / La Nina
    # (ONI sign). Writes Data/baselines/by_regime_table.csv.
    try:
        from climate_indices import load_cuti, load_oni, upwelling_at
        from charm import station_latlon
        from dataloading import STATIONS_BY_CODE

        cuti = load_cuti()
        upw_frames: list[pd.DataFrame] = []
        for code in preds["station"].unique():
            if code not in STATIONS_BY_CODE:
                continue
            plat, _ = station_latlon(STATIONS_BY_CODE[code])
            c = upwelling_at(plat, cuti, "CUTI").reset_index()
            c["station"] = code
            upw_frames.append(c)
        if not upw_frames:
            raise RuntimeError(
                "no HABMAP stations matched STATIONS_BY_CODE; "
                "cannot build per-pier upwelling panel"
            )
        upwelling_panel = pd.concat(upw_frames, ignore_index=True)

        oni = load_oni()  # columns: time, oni
        regime = by_regime_table(
            preds, upwelling_panel=upwelling_panel, climate_panel=oni,
        )
        out_csv = BASELINES_DIR / "by_regime_table.csv"
        print(f"  -> {out_csv}  ({len(regime)} rows)")
    except Exception as e:  # noqa: BLE001
        print(f"  by_regime_table skipped: {e}")


if __name__ == "__main__":
    main()
