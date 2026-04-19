"""Latency-respecting replay harness.

The single most important non-modeling piece of this project.

When you forecast at time ``t``, you may only use inputs that would
have been *available* at ``t`` in real operations. Many published
"operational" ML forecasts silently violate this -- they look up the
HABMAP weekly grab from week W when forecasting for week W, even
though that grab was processed and posted 1-4 weeks later.

The ``Latency`` table below encodes the README's documented lags:

    PACE OCI          ~24 h
    VIIRS chl-a (NRT) ~24-48 h
    MODIS L2/L3       1-3 days
    ERA5T             ~5 days
    HABMAP samples    1-4 weeks
    CDPH biotoxin     1-2 weeks
    C-HARM v3         daily, but produced from previous day's inputs
    CalCOFI bottle    ~120 days (quarterly cruise + post-processing)

Every baseline in baselines.py and every deep model added later is
expected to call ``available_inputs(t)`` to assemble its features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class Latency:
    """Real-world publish latency for one input source."""
    name: str
    delay: pd.Timedelta
    note: str = ""


# Conservative defaults. Lengthen them if your baseline turns out to be
# implausibly skillful -- almost always a sign of a latency violation.
LATENCY: Mapping[str, Latency] = {
    "pace_oci":     Latency("PACE OCI",        pd.Timedelta(hours=24)),
    "viirs_chla":   Latency("VIIRS chl-a NRT", pd.Timedelta(hours=36)),
    "modis_chla":   Latency("MODIS chl-a",     pd.Timedelta(days=2)),
    "sst_oisst":    Latency("NOAA OISST",      pd.Timedelta(days=1)),
    "era5":         Latency("ERA5T",           pd.Timedelta(days=5)),
    "glorys_nrt":   Latency("GLORYS12 NRT",    pd.Timedelta(days=10),
                            "myint product; full reanalysis lags ~3 mo"),
    "cuti":         Latency("CUTI/BEUTI",      pd.Timedelta(days=2)),
    # Per-index publication lags (audit medium item: per-index climate
    # latency). NOAA CPC posts ONI early in the following month; PSL
    # PDO and MEI.v2 typically refresh mid-month; Di Lorenzo NPGO posts
    # roughly a month after the calendar month closes. ``climate_idx``
    # remains as the conservative default for callers that have not
    # been refactored to per-index keys; values were widened from
    # 10 days to 15 days so the back-compat path is closer to the
    # mid-month publish reality of PDO / MEI.
    "oni":          Latency("ONI",       pd.Timedelta(days=10),
                            "NOAA CPC posts finalized ONI ~early next month"),
    "pdo":          Latency("PDO",       pd.Timedelta(days=15),
                            "NOAA PSL ERSSTv5 PDO refresh mid-month"),
    "mei":          Latency("MEI.v2",    pd.Timedelta(days=15),
                            "bimonthly window centered on 15th, posted shortly after"),
    "npgo":         Latency("NPGO",      pd.Timedelta(days=30),
                            "Di Lorenzo monthly NPGO is typically posted ~1 mo after the calendar month"),
    "climate_idx":  Latency("ONI/PDO/NPGO/MEI", pd.Timedelta(days=15),
                            "conservative monthly index lag; per-index keys override"),
    "habmap":       Latency("SCCOOS HABMAP",   pd.Timedelta(weeks=2),
                            "use 1-week typical, 4-week worst case"),
    "cdph_biotox":  Latency("CDPH biotoxin",   pd.Timedelta(weeks=2)),
    "charm_v3":     Latency("C-HARM v3",       pd.Timedelta(days=1)),
    "calcofi":      Latency("CalCOFI bottle",  pd.Timedelta(days=120),
                            "quarterly cruise + 2-4 mo post-processing; "
                            "climatology features built from this source "
                            "are static and ignore the lag, but any "
                            "'most recent CalCOFI cast' feature is "
                            "treated as a 4-month-lagged observation"),
}


def cutoff(source: str, forecast_time: pd.Timestamp) -> pd.Timestamp:
    """Latest sample time visible at ``forecast_time`` for one source."""
    if source not in LATENCY:
        raise KeyError(f"Unknown source {source!r}; add it to LATENCY first.")
    return forecast_time - LATENCY[source].delay


def slice_available(
    df: pd.DataFrame,
    *,
    source: str,
    forecast_time: pd.Timestamp,
    time_col: str = "time",
) -> pd.DataFrame:
    """Return only the rows of ``df`` that would have been visible at
    ``forecast_time`` given ``source``'s publish latency."""
    if df.empty:
        return df
    visible = df[df[time_col] <= cutoff(source, forecast_time)]
    return visible


def available_inputs(
    forecast_time: pd.Timestamp,
    *,
    sources: Mapping[str, pd.DataFrame],
    source_to_latency: Mapping[str, str] | None = None,
    time_col: str = "time",
) -> dict[str, pd.DataFrame]:
    """Apply the right latency cutoff to every source.

    ``sources`` is e.g. ``{"habmap": df_pier, "cuti": cuti_df, ...}``.
    The keys are normally the latency keys themselves; pass
    ``source_to_latency`` if you want to override (e.g. you have two
    HABMAP variants, both subject to the same lag).
    """
    mapping = source_to_latency or {k: k for k in sources}
    out: dict[str, pd.DataFrame] = {}
    for key, df in sources.items():
        latency_key = mapping.get(key, key)
        out[key] = slice_available(
            df, source=latency_key, forecast_time=forecast_time, time_col=time_col,
        )
    return out


def rolling_origin(
    start: str | pd.Timestamp,
    stop: str | pd.Timestamp,
    *,
    step: pd.Timedelta = pd.Timedelta(days=7),
    tz_naive: bool = True,
):
    """Yield forecast issue-times for a rolling-origin replay.

    Use this to drive 2023-2025 evaluation in baselines.py / evaluate.py.

    Returns tz-naive UTC timestamps by default to match the rest of the
    pipeline (every ``merge_asof`` consumer strips tz; mixing tz-aware
    rolling-origin times with tz-naive feature times raises
    ``TypeError`` inside merge_asof). Pass ``tz_naive=False`` only when
    interoperating with code that intentionally keeps the UTC label.
    """
    t = pd.Timestamp(start, tz="UTC")
    end = pd.Timestamp(stop, tz="UTC")
    while t <= end:
        yield t.tz_convert("UTC").tz_localize(None) if tz_naive else t
        t = t + step


if __name__ == "__main__":
    t = pd.Timestamp("2024-09-01", tz="UTC")
    print(f"Forecast issue time: {t.isoformat()}")
    for k, lat in LATENCY.items():
        print(f"  {k:13s}  cutoff = {cutoff(k, t).isoformat()}   ({lat.delay})")
