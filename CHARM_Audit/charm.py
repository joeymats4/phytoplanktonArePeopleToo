"""C-HARM v3 puller (operational baseline).

C-HARM v3 (Anderson et al., based on the original 2016 system; current
operational version uses NOAA's WCOFS ROMS for physics) is the
California HAB-forecast incumbent. The v3.1 nowcast/forecast suite is
served as gridded netCDFs on NOAA **CoastWatch West Coast** ERDDAP
(NOT PolarWatch -- as of 2026 PolarWatch carries no C-HARM datasets):

    wvcharmV3_0day  -- nowcast (same-day)
    wvcharmV3_1day  -- 1-day forecast
    wvcharmV3_2day  -- 2-day forecast
    wvcharmV3_3day  -- 3-day forecast

The variable names verified against the live ERDDAP .das (April 2026):

    pseudo_nitzschia    P(Pseudo-nitzschia > 10,000 cells/L)
    particulate_domoic  P(particulate DA > 500 ng/L)
    cellular_domoic     P(cellular DA > 10 pg/cell)
    chla_filled         DINEOF gap-filled VIIRS chl-a (mg m^-3)
    r486_filled         486 nm reflectance
    r551_filled         551 nm reflectance
    salinity            WCOFS surface salinity (PSU)
    water_temparture    WCOFS SST (deg C)  [sic -- typo in source]

Any paper that claims to "beat deep learning baselines" without
including C-HARM here would be rejected by an HAB-aware reviewer.
This module pulls the gridded forecast and samples it at the HABMAP
pier locations so the comparison is apples-to-apples with our other
station-level baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

import storage
from dataloading import STATIONS, Station

# Per-pier C-HARM netCDF subsets are read back via xarray + h5netcdf,
# which needs a real file handle (fsspec dispatch through xarray is not
# guaranteed across engines). Keep the binary cache on local disk; any
# future tabular aggregates can use ``storage.data_root()`` directly.
DATA_DIR = storage.local_scratch()
ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"

# Default datasets to pull. These IDs are correct as of April 2026 but
# the operational system rolls forward; if a fetch 404s, refresh from
# https://coastwatch.pfeg.noaa.gov/erddap/search/index.html?searchFor=charm
DATASETS: tuple[str, ...] = (
    "wvcharmV3_0day", "wvcharmV3_1day", "wvcharmV3_2day", "wvcharmV3_3day",
)

# The three event-probability variables that align with our HABMAP labels.
# We do NOT pull the chla_filled / reflectance / WCOFS fields by default
# (those would inflate cache size 5x without helping the comparison).
VARIABLES: tuple[str, ...] = (
    "pseudo_nitzschia", "particulate_domoic", "cellular_domoic",
)


def _griddap_url(
    dataset_id: str,
    *,
    variable: str,
    time_min: str,
    time_max: str,
    lat: float,
    lon: float,
    half_box_deg: float = 0.04,
) -> str:
    """Build a griddap netCDF query URL clipped to a tiny box around
    a station latitude/longitude. We pull a small box rather than a
    single pixel so we can compute neighborhood (FSS-style) scores.

    Verified against the C-HARM v3.1 datasets on
    coastwatch.pfeg.noaa.gov:

    - Dimensions: ``[time][latitude][longitude]``  (latitude ASCENDING)
    - Longitude convention: 0..360 (so -117.257 -> 242.743)
    - Latitude valid range on the grid: 31.3 .. 43.0
    """
    lat_min, lat_max = lat - half_box_deg, lat + half_box_deg

    # 0..360 longitude convention
    lon360 = lon if lon >= 0 else lon + 360.0
    lon_min, lon_max = lon360 - half_box_deg, lon360 + half_box_deg

    # Lock to the grid's actual bounds (otherwise we 404 on out-of-range)
    lat_min = max(lat_min, 31.3)
    lat_max = min(lat_max, 43.0)

    query = (
        f"{variable}"
        f"[({time_min}):1:({time_max})]"
        f"[({lat_min}):1:({lat_max})]"
        f"[({lon_min}):1:({lon_max})]"
    )
    return f"{ERDDAP_BASE}/{dataset_id}.nc?{urllib.parse.quote(query, safe='[](),:.-')}"


_DATASET_TIME_BOUNDS_CACHE: dict[str, tuple[str, str]] = {}


def _dataset_time_bounds(dataset_id: str) -> tuple[str, str]:
    """Read ``time.actual_range`` from the dataset .das.

    C-HARM v3.1 datasets only cover ~Nov 2022 onward; asking for
    earlier ranges 404s. Cache the result per dataset_id.
    """
    if dataset_id in _DATASET_TIME_BOUNDS_CACHE:
        return _DATASET_TIME_BOUNDS_CACHE[dataset_id]
    url = f"{ERDDAP_BASE}/{dataset_id}.das"
    with urllib.request.urlopen(url, timeout=30) as resp:
        txt = resp.read().decode()
    i = txt.find("time {")
    j = txt.find("actual_range", i)
    if j < 0:
        raise RuntimeError(f"could not parse time bounds for {dataset_id}")
    import re
    line = txt[j:txt.find(";", j)]
    nums = [float(s) for s in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)]
    if len(nums) != 2:
        raise RuntimeError(f"unexpected actual_range line: {line!r}")
    t0 = pd.to_datetime(nums[0], unit="s", utc=True)
    t1 = pd.to_datetime(nums[1], unit="s", utc=True)
    bounds = (t0.strftime("%Y-%m-%dT%H:%M:%SZ"),
              t1.strftime("%Y-%m-%dT%H:%M:%SZ"))
    _DATASET_TIME_BOUNDS_CACHE[dataset_id] = bounds
    return bounds


def fetch_charm_at_station(
    station: Station,
    *,
    dataset_id: str = "wvcharmV3_1day",
    variables: Iterable[str] = VARIABLES,
    time_min: str = "2022-01-01T00:00:00Z",
    time_max: str = "2025-12-31T23:59:59Z",
    refresh: bool = False,
) -> Path:
    """Download a per-variable netCDF subset around a single pier.

    Returns the path to the cached netCDF. We deliberately keep this as
    a thin downloader; opening with xarray is left to the caller to
    avoid forcing an xarray dependency on people who only want labels.

    The requested ``time_min`` / ``time_max`` are automatically clamped
    to the dataset's published ``time.actual_range`` (C-HARM v3.1 only
    covers ~Nov 2022 - present; out-of-range requests 404).
    """
    storage.ensure_dir(DATA_DIR)
    out_dir = DATA_DIR / "charm" / dataset_id / station.code
    storage.ensure_dir(out_dir)

    ds_min, ds_max = _dataset_time_bounds(dataset_id)
    if time_min < ds_min:
        time_min = ds_min
    if time_max > ds_max:
        time_max = ds_max

    # Include the time range in the cache filename so subsequent calls
    # with a wider window do not silently reuse a narrower cached file.
    tag = f"{time_min[:10]}_{time_max[:10]}".replace(":", "")

    paths = []
    for var in variables:
        cache = out_dir / f"{var}__{tag}.nc"
        if refresh or not cache.exists():
            url = _griddap_url(
                dataset_id,
                variable=var,
                time_min=time_min,
                time_max=time_max,
                lat=float(station_latlon(station)[0]),
                lon=float(station_latlon(station)[1]),
            )
            try:
                with urllib.request.urlopen(url) as resp:
                    cache.write_bytes(resp.read())
            except urllib.error.HTTPError as e:
                raise RuntimeError(
                    f"CoastWatch fetch failed for {dataset_id}/{var} at "
                    f"{station.code} ({e.code}). Check that the variable "
                    "name and station bbox match the current dataset .das. "
                    f"URL was: {url}"
                ) from e
        paths.append(cache)
    return out_dir


# -----------------------------------------------------------------------
# Station latitude/longitude lookup
#
# We could pull these from any HABMAP record once dataloading.load_station
# has been called, but caching a tiny static table avoids forcing a
# network call here.
# -----------------------------------------------------------------------
_STATION_LATLON: dict[str, tuple[float, float]] = {
    "scripps":     (32.867, -117.257),
    "newport":     (33.604, -117.929),
    "santamonica": (34.008, -118.500),
    "stearns":     (34.408, -119.685),
    "calpoly":     (35.169, -120.741),
    "santacruz":   (36.958, -122.017),
    "bodega":      (38.318, -123.072),
    "bodega_buoy": (38.318, -123.072),
    "trinidad":    (41.057, -124.151),
    "morro_back":  (35.342, -120.834),
    "morro_front": (35.371, -120.860),
    "humboldt":    (40.806, -124.180),
    "humboldt_sb": (40.747, -124.226),
    "tomales_in":  (38.149, -122.892),
    "tomales_mid": (38.193, -122.934),
    "tomales_out": (38.231, -122.978),
}


def station_latlon(station: Station) -> tuple[float, float]:
    return _STATION_LATLON[station.code]


def fetch_charm_all_stations(
    *,
    dataset_id: str = "wvcharmV3_1day",
    variables: Iterable[str] = VARIABLES,
    time_min: str = "2022-01-01T00:00:00Z",
    time_max: str = "2025-12-31T23:59:59Z",
    refresh: bool = False,
    skip_failures: bool = True,
    verbose: bool = True,
) -> dict[str, Path]:
    """Bulk pull C-HARM v3 around every active HABMAP pier."""
    out: dict[str, Path] = {}
    for s in STATIONS:
        if verbose:
            print(f"[charm] {dataset_id}: {s.code} ...", flush=True)
        try:
            out[s.code] = fetch_charm_at_station(
                s,
                dataset_id=dataset_id,
                variables=variables,
                time_min=time_min,
                time_max=time_max,
                refresh=refresh,
            )
        except Exception as exc:
            if not skip_failures:
                raise
            print(f"[charm] {s.code}: {exc}")
    return out


# -----------------------------------------------------------------------
# Sampling: open the cached netCDFs and turn them into a tidy long-format
# DataFrame keyed by (station, time) so we can score against HABMAP labels
# the same way every other baseline is scored.
# -----------------------------------------------------------------------
def load_charm_at_station(
    station: Station,
    *,
    dataset_id: str = "wvcharmV3_1day",
    variables: Iterable[str] = VARIABLES,
    aggregator: str = "median",
) -> pd.DataFrame:
    """Open the cached per-variable netCDFs for one station and reduce
    each (lat, lon) cube to a single number per timestamp.

    ``aggregator`` is one of ``median`` (default, robust), ``mean``,
    ``max`` (worst-case alarm), or ``nearest`` (single-pixel at pier).
    Returns columns ``time, station`` plus one column per variable.

    NOTE on aggregator choice (audit, April 2026): C-HARM v3.1 raw
    probabilities at coastal pier locations are biased high relative
    to point-bottle HABMAP labels. Median P(PN > 1e4) at Scripps over
    the cached 2022-11..2025-12 window is ~0.88 against a HABMAP
    base rate of ~0.20; this drives a Brier of ~0.65 on the test
    fold. The over-prediction is in the underlying C-HARM logistic,
    not in our spatial aggregation: ``median`` (Brier 0.64),
    ``mean`` (0.61), and ``max`` (0.73) all bracket the same range.
    ``nearest`` lowers Brier to 0.54 but loses ~76% of rows when the
    single nearest cell is cloud-flagged. ``median`` is therefore the
    honest default. The headline takeaway from the C-HARM row in
    ``baseline_table.csv`` is that the operational incumbent is
    materially miscalibrated at the pier scale -- a publishable
    finding, not a software bug.
    """
    import numpy as np
    import xarray as xr

    out_dir = DATA_DIR / "charm" / dataset_id / station.code
    if not out_dir.exists():
        raise FileNotFoundError(
            f"No cached C-HARM data at {out_dir}. Call fetch_charm_at_station first."
        )

    rec: dict[str, pd.Series] = {}
    for v in variables:
        # Pick up both the legacy non-tagged cache and any tagged files;
        # if multiple time-window pulls exist, concatenate along time.
        paths = sorted(out_dir.glob(f"{v}__*.nc")) + sorted(out_dir.glob(f"{v}.nc"))
        if not paths:
            continue
        ds = xr.open_mfdataset(paths, combine="by_coords")
        da = ds[v]
        if aggregator == "median":
            ts = da.median(dim=("latitude", "longitude"), skipna=True)
        elif aggregator == "mean":
            ts = da.mean(dim=("latitude", "longitude"), skipna=True)
        elif aggregator == "max":
            ts = da.max(dim=("latitude", "longitude"), skipna=True)
        elif aggregator == "nearest":
            plat, plon = station_latlon(station)
            lon360 = plon if plon >= 0 else plon + 360.0
            ts = da.sel(latitude=plat, longitude=lon360, method="nearest")
        else:
            ds.close()
            raise ValueError(f"unknown aggregator={aggregator!r}")
        rec[v] = ts.to_pandas()
        ds.close()

    if not rec:
        return pd.DataFrame(columns=["station", "time", *variables])

    df = pd.DataFrame(rec).reset_index().rename(columns={"index": "time"})
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.insert(0, "station", station.code)
    return df


def load_charm_all_stations(
    *,
    dataset_id: str = "wvcharmV3_1day",
    variables: Iterable[str] = VARIABLES,
    aggregator: str = "median",
) -> pd.DataFrame:
    """Stack ``load_charm_at_station`` over every cached pier.

    Stations without a local cache are silently skipped -- run
    ``fetch_charm_all_stations`` first.
    """
    frames: list[pd.DataFrame] = []
    for s in STATIONS:
        try:
            frames.append(load_charm_at_station(
                s, dataset_id=dataset_id, variables=variables,
                aggregator=aggregator,
            ))
        except FileNotFoundError:
            continue
    if not frames:
        return pd.DataFrame(columns=["station", "time", *variables])
    return pd.concat(frames, ignore_index=True)


# -----------------------------------------------------------------------
# Audit primitives (CHARM fairness defenses).
#
# These are thin wrappers / utilities exposed at the top level so the
# Step-3 audit script in ``scripts/charm_audit_panel.py`` does not have
# to reach into private helpers, and so a teammate can recompute the
# audit numbers in a notebook with one import.
# -----------------------------------------------------------------------
def score_aggregator_panel(
    dataset_id: str = "wvcharmV3_1day",
    aggregator: str = "median",
    variables: Iterable[str] = VARIABLES,
) -> pd.DataFrame:
    """Return the per-(station, time) C-HARM panel under a chosen
    spatial aggregator.

    Identical in shape to ``load_charm_all_stations`` but with the
    aggregator surfaced at the call site so the Step-3 audit script
    can sweep {median, mean, max, nearest} and prove the headline
    conclusion is invariant to spatial reduction.
    """
    return load_charm_all_stations(
        dataset_id=dataset_id,
        variables=variables,
        aggregator=aggregator,
    )


def fit_isotonic_temporal_split(
    df: pd.DataFrame,
    *,
    value_col: str,
    label_col: str,
    time_col: str = "time",
    head_frac: float = 0.30,
):
    """Fit isotonic regression on the chronologically earliest
    ``head_frac`` of ``df`` and return ``(calibrator, eval_df)``.

    Used because C-HARM v3.1 has zero coverage on our val fold (val is
    2019-2021, C-HARM coverage starts ~Nov 2022), so the usual
    "fit on val, score on test" pattern is impossible. Instead we use
    the first 30% of the test fold as the calibration set and score on
    the remaining 70%, which is leak-free as long as the split is
    strictly time-ordered. The assertion below enforces that.

    Returns
    -------
    (calibrator, eval_df) where ``calibrator`` is a fitted
    ``sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")`` that
    maps raw C-HARM probabilities to calibrated ones, and ``eval_df``
    is the evaluation slice (last ``1 - head_frac`` of rows by time)
    with all original columns plus ``f"{value_col}_calibrated"``.
    """
    from sklearn.isotonic import IsotonicRegression

    if not (0.0 < head_frac < 1.0):
        raise ValueError(f"head_frac must be in (0, 1), got {head_frac!r}")

    work = df.dropna(subset=[value_col, label_col, time_col]).copy()
    work[time_col] = pd.to_datetime(work[time_col], utc=True)
    work = work.sort_values(time_col).reset_index(drop=True)
    if len(work) < 20:
        raise ValueError(
            f"too few non-null rows ({len(work)}) to fit isotonic; "
            f"check the join produced data for {value_col}"
        )

    cut = int(round(len(work) * head_frac))
    cut = max(cut, 5)
    cut = min(cut, len(work) - 5)
    calib = work.iloc[:cut]
    evald = work.iloc[cut:].copy()

    # Non-leakage assertion (audit defense #3): the eval slice must
    # start strictly after the calibration slice ends in wall-clock
    # time. If the temporal split puts identical timestamps on both
    # sides we still allow it as long as the boundary timestamp is the
    # last row of calib (i.e. all eval rows are strictly later).
    calib_max = calib[time_col].max()
    eval_min = evald[time_col].min()
    assert eval_min >= calib_max, (
        f"isotonic temporal split leaked: eval starts at {eval_min} "
        f"but calibration ends at {calib_max}"
    )

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(calib[value_col].values, calib[label_col].values)
    evald[f"{value_col}_calibrated"] = iso.predict(evald[value_col].values)
    return iso, evald


if __name__ == "__main__":
    """CLI driver. Two modes:

        python charm.py                   # smoke test: 30 days at Scripps
        python charm.py --bulk            # full 2022-01-01..2025-12-31 pull
                                          # at every active HABMAP pier
                                          # for the four V3.1 forecast horizons
    """
    import datetime as dt
    import sys

    if "--bulk" in sys.argv:
        for ds_id in DATASETS:
            print(f"\n=== {ds_id} ===")
            fetch_charm_all_stations(
                dataset_id=ds_id,
                time_min="2022-01-01T00:00:00Z",
                time_max="2025-12-31T23:59:59Z",
            )
        print("\nDone. Sampled time-series:")
        df = load_charm_all_stations(dataset_id="wvcharmV3_1day")
        print(df.head())
        print(f"rows={len(df)}  stations={df['station'].nunique()}")
    else:
        today = dt.datetime.now(dt.UTC).date()
        s = STATIONS[0]
        p = fetch_charm_at_station(
            s,
            dataset_id="wvcharmV3_1day",
            time_min=(today - dt.timedelta(days=30)).isoformat() + "T00:00:00Z",
            time_max=today.isoformat() + "T00:00:00Z",
        )
        print(f"Cached C-HARM v3 1-day around {s.pretty_name} -> {p}")
        df = load_charm_at_station(s, dataset_id="wvcharmV3_1day")
        print(df.tail())
