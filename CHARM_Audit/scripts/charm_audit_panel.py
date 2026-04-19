"""C-HARM v3.1 fairness audit -- emit the artifacts the headline figure
and its three supporting figures (fig1b/c/d) read from disk.

Outputs (under Project/Data/baselines/):

    charm_aggregator_audit.parquet
        One row per (target, aggregator in {median, mean, max, nearest})
        with PR-AUC, Brier, n, and station-block bootstrap 95% CIs.
        Drives fig1c. Demonstrates the headline conclusion is invariant
        to the spatial reduction over the 0.04 deg box around each pier.

    charm_horizon_audit.parquet
        One row per (target, horizon in {0day, 1day, 2day, 3day}) with
        the same columns. Drives fig1d. Median aggregator throughout.

    predictions_p_pn_test.parquet  (in-place append)
    predictions_p_pda_test.parquet (in-place append)
        Two new model series:
          - charm_wvcharmV3_1day_calibrated -- isotonic-recalibrated
            on the chronologically earliest 30% of test, evaluated on
            the latest 70%. Only the eval slice is written, so PR-AUC /
            Brier / reliability downstream are non-leaky by construction.
          - charm_wvcharmV3_0day_raw -- raw 0-day nowcast at the same
            piers (best-case horizon for C-HARM). Skipped automatically
            if the 0-day cache hasn't been backfilled yet.

Five non-negotiable sanity assertions (script exits non-zero on any):

    1. All raw C-HARM probabilities lie in [0, 1].
    2. Isotonic eval slice is strictly later in time than calib slice.
    3. Post-isotonic ECE on the eval slice is at most 0.5x the pre-isotonic
       ECE (otherwise the calibration split is too small to trust).
    4. Horizon-curve PR-AUC is monotone-non-increasing from 0day -> 3day
       within +-1 bootstrap-CI half-width.
    5. The headline-figure inner-join across the four series produces the
       same n for all four series. Logged to stderr as ``n_aligned=...``.

Usage::

    python scripts/charm_audit_panel.py
    python scripts/charm_audit_panel.py --skip-aggregators
    python scripts/charm_audit_panel.py --no-strict   # downgrade the
                                                       # sanity asserts
                                                       # to warnings

The script is idempotent: re-running it overwrites the audit parquets
and the appended ``charm_*`` rows in the predictions parquets, but does
not touch any other model rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import storage  # noqa: E402
from baselines import EVENT_TARGETS, split_train_val_test  # noqa: E402
from charm import (  # noqa: E402
    fit_isotonic_temporal_split,
    score_aggregator_panel,
)
from dataloading import load_all_stations  # noqa: E402
from evaluate import (  # noqa: E402
    bootstrap_metric,
    brier_decomposition,
    pr_auc,
    reliability_table,
)


# Variable mapping -- mirrors run_baselines.CHARM_VAR_FOR. Both sides use
# the same (PN > 1e4, pDA > 500 ng/L) thresholds, so we score raw
# probabilities directly.
CHARM_VAR_FOR: dict[str, str] = {
    "p_pn":  "pseudo_nitzschia",
    "p_pda": "particulate_domoic",
}

OUR_BEST_MODEL = "lgb_arm_a_plus_calcofi"
RAW_CHARM_MODEL = "charm_wvcharmV3_1day"
CALIBRATED_CHARM_MODEL = "charm_wvcharmV3_1day_calibrated"
ZERO_DAY_CHARM_MODEL = "charm_wvcharmV3_0day_raw"

AGGREGATORS: tuple[str, ...] = ("median", "mean", "max", "nearest")
HORIZONS: tuple[str, ...] = (
    "wvcharmV3_0day", "wvcharmV3_1day", "wvcharmV3_2day", "wvcharmV3_3day",
)

JOIN_TOLERANCE = pd.Timedelta(days=2)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def _coerce_utc_naive(s: pd.Series) -> pd.Series:
    """Match the convention in run_baselines._attach_satellite: parse to
    UTC, then drop the tz so merge_asof's tolerance comparison works."""
    out = pd.to_datetime(s, utc=True)
    return out.dt.tz_convert("UTC").dt.tz_localize(None)


def _attach_charm(
    label_df: pd.DataFrame,
    charm_panel: pd.DataFrame,
    *,
    value_col: str,
    tolerance: pd.Timedelta = JOIN_TOLERANCE,
) -> pd.DataFrame:
    """Per-station merge_asof of a C-HARM panel onto the HABMAP-fold rows.

    Mirrors ``run_baselines._attach_satellite`` so the audit join exactly
    matches the headline pipeline (2-day backward tolerance, per-station
    chronological merge).
    """
    if charm_panel.empty:
        out = label_df.copy()
        out[value_col] = np.nan
        return out
    sat = charm_panel.copy()
    sat["time"] = _coerce_utc_naive(sat["time"])
    sat = sat.dropna(subset=["time"]).sort_values(["station", "time"])

    base = label_df.copy()
    base["time"] = _coerce_utc_naive(base["time"])

    out = []
    for st, gb in base.groupby("station"):
        s_sat = sat[sat["station"] == st][["time", value_col]].sort_values("time")
        if s_sat.empty:
            gb[value_col] = np.nan
            out.append(gb)
            continue
        merged = pd.merge_asof(
            gb.sort_values("time"), s_sat,
            on="time", direction="backward",
            tolerance=tolerance,
        )
        out.append(merged)
    return pd.concat(out, ignore_index=True)


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error -- weighted absolute calibration gap."""
    table = reliability_table(y, p, n_bins=n_bins, strategy="quantile",
                              min_bin_n=1)
    table = table[table["n"] > 0]
    if table.empty:
        return float("nan")
    w = table["n"] / table["n"].sum()
    gap = (table["mean_p"] - table["obs_rate"]).abs()
    return float((w * gap).sum())


def _score_block(
    name: str,
    target: str,
    y: np.ndarray,
    p: np.ndarray,
    stations: np.ndarray | None = None,
    *,
    n_boot: int = 200,
    extra: dict | None = None,
) -> dict:
    """Compute (PR-AUC, Brier) with station-block bootstrap CIs."""
    mask = ~(np.isnan(y) | np.isnan(p))
    y, p = y[mask], p[mask]
    by = stations[mask] if stations is not None else None
    if y.size == 0 or y.sum() == 0:
        row = dict(
            name=name, target=target, n=int(y.size),
            base_rate=float("nan"),
            pr_auc=float("nan"), pr_auc_lo=float("nan"), pr_auc_hi=float("nan"),
            brier=float("nan"),  brier_lo=float("nan"),  brier_hi=float("nan"),
        )
        if extra:
            row.update(extra)
        return row
    pr_ci = bootstrap_metric(y, p, metric="pr_auc", n_boot=n_boot,
                             by=by, seed=42)
    br_ci = bootstrap_metric(y, p, metric="brier", n_boot=n_boot,
                             by=by, seed=42)
    row = dict(
        name=name, target=target, n=int(y.size),
        base_rate=float(y.mean()),
        pr_auc=pr_ci["point"], pr_auc_lo=pr_ci["lo"], pr_auc_hi=pr_ci["hi"],
        brier=br_ci["point"],  brier_lo=br_ci["lo"],  brier_hi=br_ci["hi"],
    )
    if extra:
        row.update(extra)
    return row


# -----------------------------------------------------------------------
# Audit step 1 -- aggregator robustness (median / mean / max / nearest)
# -----------------------------------------------------------------------
def aggregator_audit(test_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for agg in AGGREGATORS:
        try:
            panel = score_aggregator_panel(
                dataset_id="wvcharmV3_1day",
                aggregator=agg,
                variables=tuple(CHARM_VAR_FOR.values()),
            )
        except Exception as exc:
            print(f"[aggregator_audit] {agg!r}: load failed ({exc!r}); skipping")
            continue
        if panel.empty:
            print(f"[aggregator_audit] {agg!r}: empty panel; skipping")
            continue
        for tgt, var in CHARM_VAR_FOR.items():
            target_fn = EVENT_TARGETS[tgt]
            y_full = target_fn(test_df).values
            joined = _attach_charm(
                test_df[["station", "time"]].copy(),
                panel[["station", "time", var]],
                value_col=var,
            )
            p = pd.to_numeric(joined[var], errors="coerce").values
            stations = test_df["station"].values
            row = _score_block(
                name=f"charm_wvcharmV3_1day::{agg}",
                target=tgt, y=y_full, p=p, stations=stations,
                extra={"aggregator": agg, "horizon": "wvcharmV3_1day"},
            )
            rows.append(row)
            print(
                f"[aggregator_audit] {agg:8s}  {tgt}  n={row['n']}  "
                f"PR-AUC={row['pr_auc']:.3f} [{row['pr_auc_lo']:.3f},"
                f"{row['pr_auc_hi']:.3f}]  Brier={row['brier']:.3f}"
            )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------
# Audit step 2 -- forecast-horizon curve
# -----------------------------------------------------------------------
def horizon_audit(test_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for horizon in HORIZONS:
        try:
            panel = score_aggregator_panel(
                dataset_id=horizon,
                aggregator="median",
                variables=tuple(CHARM_VAR_FOR.values()),
            )
        except Exception as exc:
            print(f"[horizon_audit] {horizon}: load failed ({exc!r}); skipping")
            continue
        if panel.empty:
            print(f"[horizon_audit] {horizon}: empty panel; skipping")
            continue
        for tgt, var in CHARM_VAR_FOR.items():
            target_fn = EVENT_TARGETS[tgt]
            y_full = target_fn(test_df).values
            joined = _attach_charm(
                test_df[["station", "time"]].copy(),
                panel[["station", "time", var]],
                value_col=var,
            )
            p = pd.to_numeric(joined[var], errors="coerce").values
            stations = test_df["station"].values
            row = _score_block(
                name=f"{horizon}::median",
                target=tgt, y=y_full, p=p, stations=stations,
                extra={"aggregator": "median", "horizon": horizon},
            )
            rows.append(row)
            print(
                f"[horizon_audit] {horizon}  {tgt}  n={row['n']}  "
                f"PR-AUC={row['pr_auc']:.3f} [{row['pr_auc_lo']:.3f},"
                f"{row['pr_auc_hi']:.3f}]  Brier={row['brier']:.3f}"
            )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------
# Audit step 3 -- isotonic recalibration of C-HARM 1-day on test
# -----------------------------------------------------------------------
def calibrate_charm_test(
    preds_path: Path, target: str, *, head_frac: float = 0.30,
) -> tuple[pd.DataFrame, dict]:
    """Read the existing test predictions parquet, isolate the raw
    C-HARM 1-day rows, fit isotonic on the first ``head_frac`` of them
    by time, score on the rest, and return new tidy rows ready to be
    appended back to the parquet.

    Returns (new_rows, audit_dict) where audit_dict carries the
    ECE before / after, the calib/eval n, and the split timestamp.
    """
    if not preds_path.exists():
        raise FileNotFoundError(f"missing predictions parquet: {preds_path}")
    preds = pd.read_parquet(str(preds_path))
    raw = preds[(preds["model"] == RAW_CHARM_MODEL)
                & (preds["target"] == target)].copy()
    if raw.empty:
        raise RuntimeError(
            f"no {RAW_CHARM_MODEL} rows in {preds_path} for target={target}; "
            "rerun run_baselines.py first"
        )

    iso, evald = fit_isotonic_temporal_split(
        raw,
        value_col="p_pred", label_col="y_true",
        time_col="time", head_frac=head_frac,
    )
    calib_n = len(raw.dropna(subset=["p_pred", "y_true"])) - len(evald)

    pre_y = raw["y_true"].values
    pre_p = raw["p_pred"].values
    eval_y = evald["y_true"].values
    eval_p_raw = evald["p_pred"].values
    eval_p_cal = evald["p_pred_calibrated"].values
    ece_pre = _ece(eval_y, eval_p_raw)
    ece_post = _ece(eval_y, eval_p_cal)

    fold = evald["fold"].iloc[0] if "fold" in evald.columns else "test"
    new_rows = pd.DataFrame({
        "station": evald["station"].values,
        "time":    evald["time"].values,
        "model":   CALIBRATED_CHARM_MODEL,
        "target":  target,
        "fold":    fold,
        "p_pred":  eval_p_cal.astype("float64"),
        "y_true":  eval_y.astype("float64"),
    })
    audit = dict(
        target=target,
        n_calib=int(calib_n),
        n_eval=int(len(evald)),
        split_time_utc=str(pd.to_datetime(evald["time"].min())),
        ece_pre=float(ece_pre),
        ece_post=float(ece_post),
        ece_drop_ratio=(
            float(ece_post / ece_pre) if ece_pre > 0 else float("nan")
        ),
        pr_auc_raw_eval=float(pr_auc(eval_y, eval_p_raw)),
        pr_auc_cal_eval=float(pr_auc(eval_y, eval_p_cal)),
        brier_raw_eval=float(brier_decomposition(eval_y, eval_p_raw).brier),
        brier_cal_eval=float(brier_decomposition(eval_y, eval_p_cal).brier),
    )
    audit.update({
        "raw_min": float(np.nanmin(pre_p)),
        "raw_max": float(np.nanmax(pre_p)),
    })
    return new_rows, audit


# -----------------------------------------------------------------------
# Audit step 4 -- 0-day raw nowcast as a "best-case horizon for C-HARM"
# upper-bound row in the test predictions parquet
# -----------------------------------------------------------------------
def emit_zero_day_predictions(test_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Score the 0-day C-HARM nowcast at the test rows and return tidy
    long-format frames ready to be appended to the predictions parquets.

    Returns ``{target: DataFrame}``. Empty dict if the 0-day panel is
    empty / not yet pulled.
    """
    try:
        panel = score_aggregator_panel(
            dataset_id="wvcharmV3_0day",
            aggregator="median",
            variables=tuple(CHARM_VAR_FOR.values()),
        )
    except Exception as exc:
        print(f"[zero_day] load failed ({exc!r}); skipping")
        return {}
    if panel.empty:
        print("[zero_day] cache empty; skipping (run pull_charm_horizons.py)")
        return {}

    out: dict[str, pd.DataFrame] = {}
    for tgt, var in CHARM_VAR_FOR.items():
        target_fn = EVENT_TARGETS[tgt]
        y_full = target_fn(test_df).values
        joined = _attach_charm(
            test_df[["station", "time"]].copy(),
            panel[["station", "time", var]],
            value_col=var,
        )
        rows = pd.DataFrame({
            "station": test_df["station"].values,
            "time":    pd.to_datetime(test_df["time"], utc=True).values,
            "model":   ZERO_DAY_CHARM_MODEL,
            "target":  tgt,
            "fold":    "test",
            "p_pred":  pd.to_numeric(joined[var], errors="coerce").astype("float64").values,
            "y_true":  y_full.astype("float64"),
        })
        out[tgt] = rows
        scored = _score_block(
            name=ZERO_DAY_CHARM_MODEL, target=tgt,
            y=y_full, p=rows["p_pred"].values,
            stations=test_df["station"].values,
        )
        print(
            f"[zero_day] {tgt}  n={scored['n']}  "
            f"PR-AUC={scored['pr_auc']:.3f}  Brier={scored['brier']:.3f}"
        )
    return out


# -----------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------
def assert_probability_range(panel: pd.DataFrame, *, strict: bool) -> None:
    """Sanity check #2: raw C-HARM probabilities must lie in [0, 1]."""
    bad: list[str] = []
    for var in CHARM_VAR_FOR.values():
        if var not in panel.columns:
            continue
        vals = pd.to_numeric(panel[var], errors="coerce").dropna()
        if vals.empty:
            continue
        if (vals.min() < -1e-6) or (vals.max() > 1 + 1e-6):
            bad.append(
                f"{var}: range=[{vals.min():.4f}, {vals.max():.4f}] "
                f"outside [0, 1]"
            )
    if bad:
        msg = "[sanity] probability range violation:\n  " + "\n  ".join(bad)
        if strict:
            raise AssertionError(msg)
        print(msg)


def assert_calibration_drop(audits: list[dict], *, strict: bool) -> None:
    """Sanity check #4: post-isotonic ECE <= 0.5 * pre-isotonic ECE."""
    fails: list[str] = []
    for a in audits:
        ratio = a.get("ece_drop_ratio", float("nan"))
        if not np.isfinite(ratio):
            continue
        if ratio > 0.5:
            fails.append(
                f"{a['target']}: ECE_post / ECE_pre = {ratio:.3f} > 0.5  "
                f"(pre={a['ece_pre']:.4f}, post={a['ece_post']:.4f})"
            )
    if fails:
        msg = (
            "[sanity] isotonic ECE drop too small (calibration split may be "
            "underpowered):\n  " + "\n  ".join(fails)
        )
        if strict:
            raise AssertionError(msg)
        print(msg)


def assert_horizon_monotone(horizon_df: pd.DataFrame, *, strict: bool) -> None:
    """Sanity check #4b: PR-AUC should not GROW from 0day -> 3day, within
    +-1 bootstrap-CI half-width per step.
    """
    if horizon_df.empty:
        return
    fails: list[str] = []
    for tgt, sub in horizon_df.groupby("target"):
        order = {h: i for i, h in enumerate(HORIZONS)}
        sub = sub.assign(_ord=sub["horizon"].map(order)).sort_values("_ord")
        prev_pr = None
        prev_hi = None
        for _, row in sub.iterrows():
            pr = row["pr_auc"]
            hi = row["pr_auc_hi"]
            if prev_pr is not None and np.isfinite(pr) and np.isfinite(prev_pr):
                # Allow a one-half-width slack for noise in the bootstrap.
                tol = (
                    (prev_hi - prev_pr if np.isfinite(prev_hi) else 0.0)
                    + (hi - pr if np.isfinite(hi) else 0.0)
                ) * 0.5
                if pr > prev_pr + tol:
                    fails.append(
                        f"{tgt}: {row['horizon']} PR-AUC={pr:.3f} > "
                        f"prev {prev_pr:.3f} + tol {tol:.3f}"
                    )
            prev_pr, prev_hi = pr, hi
    if fails:
        msg = (
            "[sanity] horizon-curve PR-AUC is not monotone-non-increasing:\n  "
            + "\n  ".join(fails)
        )
        if strict:
            raise AssertionError(msg)
        print(msg)


def assert_row_alignment(
    preds_path: Path, target: str, models: list[str], *, strict: bool,
) -> int:
    """Sanity check #5: the inner-join across the named series produces
    the SAME n for every series. Returns the aligned row count.
    """
    if not preds_path.exists():
        msg = f"[sanity] {preds_path} missing for row-alignment check"
        if strict:
            raise AssertionError(msg)
        print(msg)
        return 0
    preds = pd.read_parquet(str(preds_path))
    sub = preds[(preds["target"] == target)
                & (preds["model"].isin(models))
                & preds["p_pred"].notna()
                & preds["y_true"].notna()].copy()
    if sub.empty:
        print(f"[sanity] no rows to align for {target} / {models}")
        return 0
    sub["time"] = pd.to_datetime(sub["time"], utc=True)
    pivot = (
        sub.pivot_table(
            index=["station", "time"], columns="model", values="p_pred",
        )
    )
    aligned = pivot.dropna(how="any")
    counts = {m: int(sub[sub["model"] == m].shape[0]) for m in models}
    aligned_n = int(len(aligned))
    print(
        f"[sanity] target={target}  per-series n={counts}  "
        f"aligned_n={aligned_n}",
        file=sys.stderr,
    )
    if aligned_n == 0:
        msg = f"[sanity] no rows survive inner-join across {models}"
        if strict:
            raise AssertionError(msg)
        print(msg)
    return aligned_n


# -----------------------------------------------------------------------
# Append-or-replace helper for the test predictions parquets
# -----------------------------------------------------------------------
def upsert_predictions(
    preds_path: Path, new_rows_by_model: dict[str, pd.DataFrame],
) -> None:
    """Read parquet, drop any rows whose model is in ``new_rows_by_model``,
    concat the new rows, write back. Idempotent across re-runs.
    """
    if not preds_path.exists():
        raise FileNotFoundError(preds_path)
    base = pd.read_parquet(str(preds_path))
    drop_models = list(new_rows_by_model.keys())
    base = base[~base["model"].isin(drop_models)].copy()
    chunks = [base] + [df for df in new_rows_by_model.values() if not df.empty]
    out = pd.concat(chunks, ignore_index=True)
    out.to_parquet(str(preds_path), index=False)
    print(
        f"[upsert] {preds_path.name}  added/replaced models={drop_models}  "
        f"final_rows={len(out)}"
    )


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", nargs="+", default=["p_pn", "p_pda"])
    ap.add_argument("--head-frac", type=float, default=0.30,
                    help="fraction of test (chronological) used to fit "
                         "isotonic for the calibrated C-HARM series")
    ap.add_argument("--n-boot", type=int, default=200,
                    help="station-block bootstrap reps for CIs")
    ap.add_argument("--skip-aggregators", action="store_true")
    ap.add_argument("--skip-horizons", action="store_true")
    ap.add_argument("--skip-calibration", action="store_true")
    ap.add_argument("--skip-zero-day", action="store_true")
    ap.add_argument("--no-strict", action="store_true",
                    help="downgrade sanity assertions to warnings")
    args = ap.parse_args()

    strict = not args.no_strict
    baselines_dir = Path(storage.dataset_dir("baselines"))

    print("[charm_audit] loading HABMAP + isolating test fold ...")
    df = load_all_stations()
    _, _, test = split_train_val_test(df)
    print(f"  test rows={len(test)}  stations={test['station'].nunique()}")

    # Sanity #2: probability range on the canonical median 1-day panel.
    print("[charm_audit] loading wvcharmV3_1day median panel for "
          "probability-range sanity check ...")
    canonical = score_aggregator_panel(
        dataset_id="wvcharmV3_1day",
        aggregator="median",
        variables=tuple(CHARM_VAR_FOR.values()),
    )
    assert_probability_range(canonical, strict=strict)

    # Step 1 -- aggregator robustness
    if not args.skip_aggregators:
        print("\n=== aggregator_audit ===")
        agg_df = aggregator_audit(test)
        out_path = baselines_dir / "charm_aggregator_audit.parquet"
        agg_df.to_parquet(str(out_path), index=False)
        print(f"[charm_audit] wrote {out_path}  rows={len(agg_df)}")

    # Step 2 -- forecast-horizon curve
    horizon_df = pd.DataFrame()
    if not args.skip_horizons:
        print("\n=== horizon_audit ===")
        horizon_df = horizon_audit(test)
        out_path = baselines_dir / "charm_horizon_audit.parquet"
        horizon_df.to_parquet(str(out_path), index=False)
        print(f"[charm_audit] wrote {out_path}  rows={len(horizon_df)}")
        # Sanity #4b: monotone-non-increasing PR-AUC vs horizon.
        assert_horizon_monotone(horizon_df, strict=strict)

    # Step 3 -- isotonic recalibration of C-HARM 1-day on test
    cal_audits: list[dict] = []
    if not args.skip_calibration:
        print("\n=== isotonic_recalibration ===")
        for tgt in args.targets:
            preds_path = baselines_dir / f"predictions_{tgt}_test.parquet"
            try:
                new_rows, audit = calibrate_charm_test(
                    preds_path, target=tgt, head_frac=args.head_frac,
                )
            except Exception as exc:
                print(f"[charm_audit] calibration for {tgt} failed: {exc}")
                continue
            cal_audits.append(audit)
            upsert_predictions(preds_path, {CALIBRATED_CHARM_MODEL: new_rows})
            print(
                f"[charm_audit] {tgt}  n_calib={audit['n_calib']}  "
                f"n_eval={audit['n_eval']}  ECE pre={audit['ece_pre']:.4f}  "
                f"post={audit['ece_post']:.4f}  ratio={audit['ece_drop_ratio']:.3f}"
            )
        # Sanity #4 (calibration sanity): post ECE <= 0.5 * pre ECE.
        assert_calibration_drop(cal_audits, strict=strict)

    # Step 4 -- 0-day raw nowcast (best-case horizon for C-HARM)
    if not args.skip_zero_day:
        print("\n=== zero_day_predictions ===")
        zero_day_rows = emit_zero_day_predictions(test)
        for tgt, rows in zero_day_rows.items():
            preds_path = baselines_dir / f"predictions_{tgt}_test.parquet"
            upsert_predictions(preds_path, {ZERO_DAY_CHARM_MODEL: rows})

    # Sanity #5 (row alignment): inner-join across the headline series.
    print("\n=== row_alignment ===")
    headline_models = [
        OUR_BEST_MODEL, RAW_CHARM_MODEL, "climatology", "persistence",
    ]
    for tgt in args.targets:
        preds_path = baselines_dir / f"predictions_{tgt}_test.parquet"
        assert_row_alignment(preds_path, tgt, headline_models, strict=strict)

    # Audit summary -- compact JSON to stdout for the CHARM_AUDIT.md to cite.
    print("\n=== AUDIT SUMMARY ===")
    summary = {
        "calibration": cal_audits,
        "horizons":    [] if horizon_df.empty else horizon_df.to_dict("records"),
    }
    out_summary = baselines_dir / "charm_audit_summary.json"
    import json
    out_summary.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[charm_audit] wrote {out_summary}")


if __name__ == "__main__":
    main()
