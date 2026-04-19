"""Baseline forecasters for the HABMAP event head.

The README is emphatic: a deep model is only the story if it beats this
hierarchy by a meaningful margin on the held-out 2022-2025 events.

Three baselines are implemented here, in order of sophistication:

    1. Climatology  -- per-station, per-week-of-year event rate.
    2. Persistence  -- "tomorrow looks like the most recent observation."
    3. LightGBM     -- gradient-boosted trees on engineered features
                       (chl-a + nutrient lags, climate indices, day-of-year
                       sin/cos, station one-hot). ~80 features routinely
                       beats vanilla LSTMs in published HAB ML work.

All three respect ``replay.available_inputs`` so the comparison against
C-HARM v3 in evaluate.py is honest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from replay import LATENCY, cutoff, slice_available

PN_THRESHOLD_CELLS_PER_L = 1e4
PDA_THRESHOLD_NG_PER_ML = 0.5  # 500 ng/L expressed in HABMAP's typical ng/mL units

# HABMAP base columns that are observed concurrently with the label and would
# therefore be subject to the same SCCOOS publish lag (~1-4 wk). We MUST NOT
# expose any of these as a feature at the row's own sample time -- doing so
# is the README's explicit "operational replay" violation. Lagged versions
# computed inside ``_make_features`` are fine because they correspond to
# *previous* HABMAP grabs.
_HABMAP_BASE_COLS: tuple[str, ...] = (
    "Avg_Chloro", "Chl1", "Chl2", "Avg_Phaeo", "Phaeo1", "Phaeo2",
    "Temp", "Air_Temp", "Salinity",
    "Phosphate", "Silicate", "Nitrate", "Nitrite", "Nitrite_Nitrate", "Ammonium",
    "DA_Volume_Filtered", "pDA", "tDA", "dDA",
    "Volume_Settled_for_Counting", "Chl_Volume_Filtered",
    "Akashiwo_sanguinea", "Alexandrium_spp", "Dinophysis_spp",
    "Lingulodinium_polyedra", "Prorocentrum_spp",
    "Pseudo_nitzschia_delicatissima_group",
    "Pseudo_nitzschia_seriata_group",
    "Ceratium_spp", "Cochlodinium_spp", "Gymnodinium_spp",
    "Other_Diatoms", "Other_Dinoflagellates", "Total_Phytoplankton",
    # geometry / bookkeeping that occasionally shows up in HABMAP CSVs
    "depth", "latitude", "longitude",
    # derived but still concurrent with the label
    "log_chla",
)


# -----------------------------------------------------------------------
# Target callables (NaN-aware)
#
# README convention: PN event = (P. seriata + P. delicatissima) > 1e4 cells/L
# so the comparison against C-HARM v3.1's ``pseudo_nitzschia`` variable
# (total PN) is apples-to-apples. The summing is NaN-safe via min_count=1
# (NaN+NaN -> NaN; NaN+x -> x), and rows where the entire PN total is
# NaN return NaN labels so they get dropped from the train/eval frame
# rather than silently treated as "non-event".
# -----------------------------------------------------------------------
def _pn_total(df: pd.DataFrame) -> pd.Series:
    cols = [
        c for c in ("Pseudo_nitzschia_seriata_group",
                    "Pseudo_nitzschia_delicatissima_group")
        if c in df.columns
    ]
    if not cols:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return df[cols].astype(float).sum(axis=1, min_count=1)


def _binary_event(values: pd.Series, threshold: float) -> pd.Series:
    out = pd.Series(np.nan, index=values.index, dtype=float)
    mask = values.notna()
    out.loc[mask] = (values.loc[mask] > threshold).astype(float)
    return out


def _pn_event(df: pd.DataFrame) -> pd.Series:
    return _binary_event(_pn_total(df), PN_THRESHOLD_CELLS_PER_L)


def _pda_event(df: pd.DataFrame) -> pd.Series:
    if "pDA" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return _binary_event(df["pDA"].astype(float), PDA_THRESHOLD_NG_PER_ML)


EVENT_TARGETS: dict[str, callable] = {
    "p_pn":  _pn_event,
    "p_pda": _pda_event,
}


# -----------------------------------------------------------------------
# 1. Climatology -- per-station week-of-year event rate
# -----------------------------------------------------------------------
@dataclass
class ClimatologyBaseline:
    """Predict P(event) as the long-run rate at this station, week-of-year."""
    target_fn: callable
    rate_table: pd.DataFrame | None = None  # index=(station, woy), col="rate"

    # Fallback hierarchy when (station, woy) is missing in the rate
    # table. Used by predict() instead of the previous "global mean of
    # whatever happens to have joined" hack, which silently mixed in
    # rates from other stations and could collapse to NaN if the entire
    # merge missed.
    by_station_: pd.Series | None = None   # index=station -> rate
    by_woy_: pd.Series | None = None       # index=woy -> rate
    overall_: float | None = None

    def fit(self, df: pd.DataFrame) -> "ClimatologyBaseline":
        d = df.copy()
        d["woy"] = d["time"].dt.isocalendar().week.astype(int)
        d["y"] = self.target_fn(d)
        d = d.dropna(subset=["y"])
        self.rate_table = (
            d.groupby(["station", "woy"])["y"].mean().rename("rate").to_frame()
        )
        self.by_station_ = d.groupby("station")["y"].mean()
        self.by_woy_ = d.groupby("woy")["y"].mean()
        self.overall_ = float(d["y"].mean()) if len(d) else 0.0
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.rate_table is None:
            raise RuntimeError("call fit() first")
        d = df.copy()
        d["woy"] = d["time"].dt.isocalendar().week.astype(int)
        joined = d.merge(self.rate_table, left_on=["station", "woy"],
                         right_index=True, how="left")
        rate = joined["rate"]
        # Fallback hierarchy: per-station mean -> per-woy mean -> overall.
        if self.by_station_ is not None:
            rate = rate.fillna(joined["station"].map(self.by_station_))
        if self.by_woy_ is not None:
            rate = rate.fillna(joined["woy"].map(self.by_woy_))
        if self.overall_ is not None:
            rate = rate.fillna(self.overall_)
        return rate


# -----------------------------------------------------------------------
# 2. Persistence -- last observed value, latency-aware
# -----------------------------------------------------------------------
@dataclass
class PersistenceBaseline:
    """At forecast time t, predict the most recent observation visible
    given HABMAP's publish lag."""
    target_fn: callable
    history: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame) -> "PersistenceBaseline":
        d = df[["station", "time"]].copy()
        d["y"] = self.target_fn(df).values
        self.history = d.dropna(subset=["y"]).sort_values("time").reset_index(drop=True)
        return self

    def predict_at(self, station: str, forecast_time: pd.Timestamp) -> float:
        if self.history is None:
            raise RuntimeError("call fit() first")
        cut = cutoff("habmap", forecast_time)
        sub = self.history[(self.history["station"] == station) & (self.history["time"] <= cut)]
        if sub.empty:
            return np.nan
        return float(sub["y"].iloc[-1])

    def predict_frame(self, query: pd.DataFrame) -> pd.Series:
        return query.apply(
            lambda r: self.predict_at(r["station"], r["time"]),
            axis=1,
        )


# -----------------------------------------------------------------------
# 3. LightGBM baseline -- engineered features, latency-aware
# -----------------------------------------------------------------------
def _make_features(
    df: pd.DataFrame,
    *,
    lag_weeks: Sequence[int] = (1, 2, 4, 8),
    climate: pd.DataFrame | None = None,   # monthly indices: time, oni, pdo, mei, npgo
    upwelling: pd.DataFrame | None = None, # daily CUTI/BEUTI by station lat
    satellite: pd.DataFrame | Sequence | None = None,
    # ``satellite`` may be either a single DataFrame (back-compat: VIIRS-style
    # daily panel ``station, time, <chla cols>``) OR an iterable of dicts of
    # the form ``{"panel": df, "value_cols": [...], "latency": "viirs_chla"}``
    # to merge multiple satellite sources independently (e.g. VIIRS chla +
    # PACE Rrs PCs, where VIIRS rows and PACE rows live on different time
    # grids and a single merge_asof would lose one of them).
    sat_value_cols: Sequence[str] | None = None,  # which sat columns to carry (single-DataFrame path)
    calcofi: pd.DataFrame | None = None,   # quarterly per-pier CalCOFI panel (calcofi.bottle_to_pier_panel / nutrient_anomaly_panel)
    calcofi_value_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build the engineered-feature table the LightGBM baseline trains on.

    Latency-respecting in the README's "operational replay" sense:

    * Every column listed in ``_HABMAP_BASE_COLS`` (the row's own HABMAP
      grab: PN, pDA, chl-a, T, S, nutrients, ...) is preserved on the
      DataFrame so ``EVENT_TARGETS`` can compute the label from it, but
      it is excluded from the feature matrix by ``_feature_columns``.
      The model sees only *lagged* HABMAP values, computed by shifting
      each per-station series ``k * 7 days`` backward in calendar time
      (so ``lag1w`` always means "1 week ago," regardless of irregular
      sampling cadence).
    * Climate / upwelling / satellite / CalCOFI joins shift the source
      ``time`` forward by the source's publish latency from
      ``replay.LATENCY``, then ``merge_asof(direction="backward")``
      so each row only sees data that would have been published by
      forecast time.

    A ``row_time`` column is added (== the original sample time, naive
    UTC). It is used by ``LightGBMBaseline.predict`` to merge the
    predicted probabilities back to the input frame -- the historical
    bug was returning predictions in ``feats`` order while labelling
    them with ``df.index``, which scrambled every score in
    ``baseline_table.csv``.
    """
    out = df.sort_values(["station", "time"]).copy()
    out["time"] = _to_naive(out["time"])
    out["row_time"] = out["time"]

    # numeric HABMAP covariates we want lagged versions of. Anything in
    # _HABMAP_BASE_COLS that the input frame actually carries.
    base_cols = [c for c in _HABMAP_BASE_COLS if c in out.columns and c != "log_chla"]

    # Time-based lag: shift the per-station series backward by k weeks of
    # calendar time, then re-align to the row's row_time. This is honest
    # for the irregular weekly cadence (some stations sample every 5 d,
    # some every 14 d). We use a backward asof on a per-station copy.
    if base_cols:
        lag_pieces: list[pd.DataFrame] = []
        for st, gb in out.groupby("station", sort=False):
            sub = gb[["row_time", *base_cols]].sort_values("row_time").copy()
            cols: dict[str, np.ndarray] = {"station": np.full(len(sub), st, dtype=object),
                                           "row_time": sub["row_time"].values}
            for k in lag_weeks:
                shifted = sub.copy()
                shifted["row_time"] = shifted["row_time"] + pd.Timedelta(days=7 * k)
                # backward asof: latest sample whose row_time + k*7d <= current row_time
                # i.e. the most recent observation at least k weeks ago.
                m = pd.merge_asof(
                    sub[["row_time"]],
                    shifted.sort_values("row_time"),
                    on="row_time",
                    direction="backward",
                )
                for col in base_cols:
                    cols[f"{col}_lag{k}w"] = m[col].values
            lag_pieces.append(pd.DataFrame(cols))
        lag_panel = pd.concat(lag_pieces, ignore_index=True)
        out = out.merge(
            lag_panel,
            on=["station", "row_time"],
            how="left",
        )

    # log chl-a (lognormal in coastal waters; see README). Kept on the
    # frame for diagnostics but EXCLUDED from features at row time.
    if "Avg_Chloro" in out.columns:
        out["log_chla"] = np.log10(out["Avg_Chloro"].clip(lower=1e-3))
        # log_chla_lag uses the already-built Avg_Chloro_lag*w; just take log.
        for k in lag_weeks:
            src = f"Avg_Chloro_lag{k}w"
            if src in out.columns:
                out[f"log_chla_lag{k}w"] = np.log10(out[src].clip(lower=1e-3))

    # day-of-year encoding
    doy = out["time"].dt.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 366.0)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 366.0)
    out["year"] = out["time"].dt.year

    # climate indices: monthly. Apply latency by shifting the join time.
    if climate is not None and not climate.empty:
        cl = climate.copy()
        cl["time"] = _to_naive(cl["time"]) + LATENCY["climate_idx"].delay
        cl = cl.sort_values("time")
        out = pd.merge_asof(
            out.sort_values("time"), cl,
            on="time", direction="backward",
        )

    # upwelling: daily. Two supported shapes:
    #   (a) global frame: time, CUTI[, BEUTI]    -- single latitude
    #   (b) per-pier:     station, time, CUTI[, BEUTI]
    # The per-pier path is what the README actually wants (CUTI varies
    # by ~5 deg lat across our pier set); the global path is kept for
    # back-compat with the older callers.
    if upwelling is not None and not upwelling.empty:
        up = upwelling.copy()
        if "time" not in up.columns:
            up = up.reset_index().rename(columns={"index": "time"})
        up["time"] = _to_naive(up["time"]) + LATENCY["cuti"].delay
        up_value_cols = [c for c in up.columns if c not in ("station", "time")]

        if "station" in up.columns:
            up = up.sort_values(["station", "time"])
            merged_up: list[pd.DataFrame] = []
            for st, gb in out.groupby("station"):
                sub_up = up[up["station"] == st][["time", *up_value_cols]] \
                    .sort_values("time")
                if sub_up.empty:
                    for c in up_value_cols:
                        gb[c] = np.nan
                    merged_up.append(gb)
                    continue
                merged_up.append(pd.merge_asof(
                    gb.sort_values("time"), sub_up,
                    on="time", direction="backward",
                    tolerance=pd.Timedelta(days=14),
                ))
            out = pd.concat(merged_up, ignore_index=True)
        else:
            up = up.sort_values("time")
            out = pd.merge_asof(
                out.sort_values("time"), up,
                on="time", direction="backward",
            )

    # satellite (per-station merge_asof; respect per-source publish latency)
    # Normalize the API: accept either a single DataFrame (back-compat) or
    # an iterable of {panel, value_cols, latency} specs for joining
    # multiple satellite sources independently.
    sat_specs: list[dict] = []
    if satellite is not None:
        if isinstance(satellite, pd.DataFrame):
            if not satellite.empty:
                sat_specs.append({
                    "panel": satellite,
                    "value_cols": list(sat_value_cols) if sat_value_cols else None,
                    "latency": "viirs_chla",
                })
        else:
            for spec in satellite:
                panel = spec.get("panel") if isinstance(spec, dict) else spec[0]
                if panel is None or panel.empty:
                    continue
                sat_specs.append({
                    "panel": panel,
                    "value_cols": (
                        spec.get("value_cols") if isinstance(spec, dict)
                        else (spec[1] if len(spec) > 1 else None)
                    ),
                    "latency": (
                        spec.get("latency", "viirs_chla") if isinstance(spec, dict)
                        else (spec[2] if len(spec) > 2 else "viirs_chla")
                    ),
                })

    for spec in sat_specs:
        sat = spec["panel"].copy()
        sat["time"] = _to_naive(sat["time"])
        sat_cols = list(spec["value_cols"]) if spec["value_cols"] else [
            c for c in sat.columns if c not in ("station", "time")
        ]
        # Apply per-source latency. PACE OCI is faster than VIIRS NRT
        # (~24h vs ~36h), but the conservative honest choice is the
        # per-source documented lag in replay.LATENCY.
        latency_key = spec["latency"]
        sat["time"] = sat["time"] + LATENCY[latency_key].delay
        sat = sat.sort_values(["station", "time"])
        merged = []
        for st, gb in out.groupby("station"):
            sub_sat = sat[sat["station"] == st][["time", *sat_cols]].sort_values("time")
            if sub_sat.empty:
                for c in sat_cols:
                    gb[c] = np.nan
                merged.append(gb)
                continue
            # 14 d tolerance comfortably covers VIIRS DAY (1 d composite +
            # 36 h latency = ~2.5 d gap) and 8D (8 d composite + 36 h
            # latency = ~9.5 d gap). Widen this to ~45 d if a monthly
            # composite is ever wired in (otherwise ~half the HABMAP rows
            # would lose their satellite feature). Stamping in
            # satellite._l3m_preprocess uses time_coverage_end so this
            # tolerance is measured from the end of each composite window.
            m = pd.merge_asof(
                gb.sort_values("time"),
                sub_sat,
                on="time", direction="backward",
                tolerance=pd.Timedelta(days=14),
            )
            merged.append(m)
        out = pd.concat(merged, ignore_index=True)
        # log of satellite chl-a (lognormal)
        for c in sat_cols:
            if "chla" in c.lower():
                out[f"log_{c}"] = np.log10(out[c].clip(lower=1e-3))

    # CalCOFI per-pier nutrient panel. Cruises are quarterly and the
    # processed bottle CSV posts ~2-4 months after the cruise; we
    # apply the calcofi latency (~120 d) so the "most-recent visible
    # cast" feature is operationally honest, and use a 180-day join
    # tolerance so we always find SOMETHING for a pier even between
    # cruises. Climatology-derived columns (`*_clim`, `*_anom`) are
    # static priors so this latency only restricts the "raw current
    # cast" columns; we keep both kinds in the panel.
    if calcofi is not None and not calcofi.empty:
        cc = calcofi.copy()
        cc["time"] = _to_naive(cc["time"])
        cc_value_cols = list(calcofi_value_cols) if calcofi_value_cols else [
            c for c in cc.columns
            if c not in ("station", "time", "sta_id", "dist_km")
            and cc[c].dtype.kind in "fiub"
        ]
        cc["time"] = cc["time"] + LATENCY["calcofi"].delay
        cc = cc.sort_values(["station", "time"])
        merged_cc: list[pd.DataFrame] = []
        for st, gb in out.groupby("station"):
            sub_cc = cc[cc["station"] == st][["time", *cc_value_cols]] \
                .sort_values("time")
            if sub_cc.empty:
                for c in cc_value_cols:
                    gb[c] = np.nan
                merged_cc.append(gb)
                continue
            merged_cc.append(pd.merge_asof(
                gb.sort_values("time"), sub_cc,
                on="time", direction="backward",
                tolerance=pd.Timedelta(days=180),
            ))
        out = pd.concat(merged_cc, ignore_index=True)

    out["station_id"] = out["station"].astype("category").cat.codes
    return out


def _to_naive(s):
    """Strip tz to UTC-naive so merge_asof works across mixed sources."""
    s = pd.to_datetime(s, utc=True, errors="coerce")
    if hasattr(s, "dt"):
        return s.dt.tz_convert("UTC").dt.tz_localize(None)
    if s.tzinfo is not None:
        return s.tz_convert("UTC").tz_localize(None)
    return s


def _feature_columns(features: pd.DataFrame) -> list[str]:
    """Pick the numeric feature columns LightGBM / the GNN train on.

    Excludes:
    * bookkeeping (station, time, row_time, Location_Code, SampleID,
      cst_cnt, btl_cnt, sta_id, dist_km),
    * the row's own HABMAP base columns (``_HABMAP_BASE_COLS``) -- these
      are the label or directly correlated with it and are unavailable
      at forecast time given the SCCOOS publish lag,
    * any synthesized helper column whose name starts with ``__``,
    * the model's own training label ``y``.
    """
    drop: set[str] = {
        "station", "time", "row_time",
        "Location_Code", "SampleID",
        "cst_cnt", "btl_cnt", "sta_id", "dist_km",
        "y",
    }
    drop |= set(_HABMAP_BASE_COLS)
    cols = [c for c in features.columns if c not in drop]
    cols = [c for c in cols if not c.startswith("__")]
    cols = [c for c in cols if features[c].dtype.kind in "fiub"]
    return cols


@dataclass
class LightGBMBaseline:
    """Per-target LightGBM head with latency-aware engineered features.

    Requires ``lightgbm`` -- install via ``pip install lightgbm``.

    Includes an optional isotonic-regression post-calibrator (the README
    section "Class imbalance" recommendation): when ``calibrate=True``
    and a ``val_df`` is supplied to ``fit``, we fit
    ``sklearn.isotonic.IsotonicRegression`` on the val-fold predicted
    probabilities and apply it inside ``predict``. This is what brings
    the Brier score below climatology on imbalanced HAB targets.
    """
    target_name: str
    target_fn: callable
    params: dict | None = None
    model: object | None = None
    feature_cols_: list[str] | None = None
    calibrator_: object | None = None
    calibrate: bool = True

    def _default_params(self) -> dict:
        # Conservative defaults: rare-event binary classification on a
        # small (~2k row) train set. Heavier regularization than the
        # LightGBM defaults; early stopping in fit() does the rest.
        #
        # Note: we deliberately do NOT use scale_pos_weight here -- it
        # multiplies the gradient on positives and produces systematically
        # over-confident probabilities, blowing up Brier even when ranking
        # (PR-AUC) is fine. We instead apply isotonic-regression
        # post-calibration on the val fold inside fit() (see calibrator_).
        return dict(
            objective="binary",
            metric="binary_logloss",
            learning_rate=0.03,
            num_leaves=15,
            min_data_in_leaf=30,
            feature_fraction=0.7,
            bagging_fraction=0.7,
            bagging_freq=5,
            lambda_l2=1.0,
            num_threads=2,  # bound joblib pool so it doesn't fight torch
            verbosity=-1,
            # Audit medium item: full RNG seeding for reproducibility.
            # LightGBM samples both rows (bagging_fraction) and columns
            # (feature_fraction); without explicit seeds each run drew
            # different subsets and the baseline_table varied between
            # processes. Single ``seed`` propagates to the per-mechanism
            # seeds when those are not set explicitly.
            seed=42,
            bagging_seed=42,
            feature_fraction_seed=42,
            data_random_seed=42,
            deterministic=True,
        )

    def fit(
        self,
        df: pd.DataFrame,
        *,
        val_df: pd.DataFrame | None = None,
        climate: pd.DataFrame | None = None,
        upwelling: pd.DataFrame | None = None,
        satellite: pd.DataFrame | Sequence | None = None,
        sat_value_cols: Sequence[str] | None = None,
        calcofi: pd.DataFrame | None = None,
        calcofi_value_cols: Sequence[str] | None = None,
        num_boost_round: int = 1500,
        early_stopping_rounds: int = 50,
    ) -> "LightGBMBaseline":
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError("LightGBM baseline needs `pip install lightgbm`") from e

        feats = _make_features(
            df, climate=climate, upwelling=upwelling,
            satellite=satellite, sat_value_cols=sat_value_cols,
            calcofi=calcofi, calcofi_value_cols=calcofi_value_cols,
        )
        feats["y"] = self.target_fn(feats).values
        feats = feats.dropna(subset=["y"])

        self.feature_cols_ = _feature_columns(feats)
        X = feats[self.feature_cols_]
        y = feats["y"].astype(int)

        train = lgb.Dataset(X, y)

        valid_sets: list = [train]
        valid_names: list[str] = ["train"]
        callbacks: list = []
        Xv = None
        yv = None
        # Audit fix (H5): always attach val for early stopping when val is
        # non-empty, even if it has zero positives. The earlier code
        # required ``vfeats["y"].sum() > 0`` for BOTH early stopping AND
        # isotonic calibration, so on degenerate val folds it silently
        # ran the full ``num_boost_round`` with no validation signal at
        # all and reverted to whatever LightGBM converged on without
        # supervision -- catastrophic when train and the underlying
        # process diverged. Now early stopping uses log-loss (which is
        # well-defined as long as val has any rows, even all-negative);
        # only isotonic calibration requires positive mass and is
        # skipped with a warning otherwise.
        if val_df is not None and not val_df.empty:
            vfeats = _make_features(
                val_df, climate=climate, upwelling=upwelling,
                satellite=satellite, sat_value_cols=sat_value_cols,
                calcofi=calcofi, calcofi_value_cols=calcofi_value_cols,
            )
            vfeats["y"] = self.target_fn(vfeats).values
            vfeats = vfeats.dropna(subset=["y"])
            if len(vfeats) > 0:
                Xv = vfeats.reindex(columns=self.feature_cols_)
                yv = vfeats["y"].astype(int)
                valid_sets.append(lgb.Dataset(Xv, yv, reference=train))
                valid_names.append("val")
                callbacks.append(lgb.early_stopping(early_stopping_rounds,
                                                   verbose=False))
                if int(yv.sum()) == 0:
                    print(
                        f"[LightGBMBaseline] target={self.target_name}: "
                        f"val has {len(yv)} rows but 0 positive events; "
                        f"early stopping uses logloss only, isotonic "
                        f"calibration will be skipped."
                    )
        callbacks.append(lgb.log_evaluation(period=0))

        self.model = lgb.train(
            self.params or self._default_params(),
            train,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        # Isotonic post-calibration on val fold. Required because LightGBM
        # binary outputs on imbalanced (~2-9% base-rate) targets are
        # systematically miscalibrated, which dominates the Brier score
        # comparison against climatology.  Skipped when val has no
        # positive events (isotonic on all-negative labels is
        # ill-defined and would collapse every prediction to 0).
        if (
            self.calibrate
            and Xv is not None and yv is not None
            and len(yv) > 0 and int(yv.sum()) > 0
        ):
            try:
                from sklearn.isotonic import IsotonicRegression
                p_val_raw = self.model.predict(Xv)
                self.calibrator_ = IsotonicRegression(
                    y_min=0.0, y_max=1.0, out_of_bounds="clip",
                ).fit(p_val_raw, yv.values)
            except Exception as e:
                # Calibration is a nice-to-have, not a hard requirement;
                # if sklearn is missing or the val fold is degenerate,
                # fall back to raw probabilities and warn.
                print(f"[LightGBMBaseline] isotonic calibration skipped: {e}")
                self.calibrator_ = None
        return self

    def predict(
        self,
        df: pd.DataFrame,
        *,
        climate: pd.DataFrame | None = None,
        upwelling: pd.DataFrame | None = None,
        satellite: pd.DataFrame | Sequence | None = None,
        sat_value_cols: Sequence[str] | None = None,
        calcofi: pd.DataFrame | None = None,
        calcofi_value_cols: Sequence[str] | None = None,
    ) -> pd.Series:
        if self.model is None or self.feature_cols_ is None:
            raise RuntimeError("call fit() first")
        feats = _make_features(
            df, climate=climate, upwelling=upwelling,
            satellite=satellite, sat_value_cols=sat_value_cols,
            calcofi=calcofi, calcofi_value_cols=calcofi_value_cols,
        )
        X = feats.reindex(columns=self.feature_cols_)
        p = self.model.predict(X)
        if self.calibrator_ is not None:
            p = self.calibrator_.transform(p)

        # Row-alignment fix: ``feats`` is in (station, time) order after
        # the per-pier sort + concat inside ``_make_features``; the input
        # ``df`` may be in any order. Merge predictions back onto df via
        # the (station, row_time) key. Historical bug: returning
        # ``pd.Series(p, index=df.index)`` glued the i-th feats prediction
        # to the i-th df.index label, scrambling every score downstream.
        pred_df = pd.DataFrame({
            "station": feats["station"].astype(str).values,
            "_pred_time": pd.to_datetime(feats["row_time"]).values,
            "_p": np.asarray(p, dtype=float),
        })
        pred_df = pred_df.drop_duplicates(
            subset=["station", "_pred_time"], keep="last",
        )

        d = df[["station", "time"]].copy()
        d["station"] = d["station"].astype(str)
        d["_pred_time"] = _to_naive(d["time"])
        d_with_p = d.merge(pred_df, on=["station", "_pred_time"], how="left")
        return pd.Series(
            d_with_p["_p"].values, index=df.index, name=self.target_name,
        )


# -----------------------------------------------------------------------
# Train/val/test split following the README convention
# -----------------------------------------------------------------------
def split_train_val_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """README split: train 2012-2018, val 2019-2021, test 2022-2025.

    2015 is held out entirely from training -- the canonical extreme-event
    year. We move it into a separate frame the caller can score on
    (``extreme_2015(df)``).

    Note: HABMAP starts in 2008, so 2008-2011 rows are deliberately
    *dropped* from train (per README's "post-VIIRS, pre-major-MHW
    aftermath" rationale). They are also not exposed to climatology /
    persistence baselines via this split; if you want climatology to
    use the full pre-2019 record, fit on ``df[df["time"].dt.year < 2019]``
    directly rather than going through ``split_train_val_test``.
    """
    yr = df["time"].dt.year
    test = df[yr.isin([2022, 2023, 2024, 2025])]
    val  = df[yr.isin([2019, 2020, 2021])]
    train = df[(yr >= 2012) & (yr <= 2018) & (yr != 2015)]
    return train, val, test


def extreme_2015(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["time"].dt.year == 2015]


if __name__ == "__main__":
    from dataloading import load_all_stations
    print("Loading stations (cached)...")
    df = load_all_stations()

    train, val, test = split_train_val_test(df)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}  "
          f"extreme-2015={len(extreme_2015(df))}")

    print("\nFitting climatology baseline (target: pDA event)...")
    clim = ClimatologyBaseline(target_fn=EVENT_TARGETS["p_pda"]).fit(train)
    pred = clim.predict(test)
    print("Mean predicted P(pDA event) on 2022-25:", float(pred.mean()))

    print("\nFitting persistence baseline (target: pDA event)...")
    pers = PersistenceBaseline(target_fn=EVENT_TARGETS["p_pda"]).fit(train)
    pred_p = pers.predict_frame(test[["station", "time"]])
    print("Persistence non-null on test:", int(pred_p.notna().sum()))
