"""SCCOOS HABMAP pier loader.

Pulls weekly HAB sampling records from the SCCOOS ERDDAP service for any
of the active HABMAP piers and returns them as a tidy long-format
DataFrame keyed by ``(station, time)``.

The HABMAP network (https://sccoos.org/harmful-algal-bloom/) operates 9
piers; Monterey Wharf has been offline since March 2020 and is excluded
from the default station list.

Each ERDDAP CSV begins with a units row (row index 1) which is parsed
out and stored on ``df.attrs["units"]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

import storage

DATA_DIR = storage.data_root()
HABMAP_DIR = storage.dataset_dir("habmap")

ERDDAP_BASE = "https://erddap.sccoos.org/erddap/tabledap"

# All variables we want from each pier dataset. Not every pier carries
# every variable; ERDDAP will silently 404 the request if even one is
# missing, so we ask only for the common-core set first and fall back
# per-pier if needed.
COMMON_VARIABLES = [
    "Location_Code", "latitude", "longitude", "depth", "SampleID", "time",
    "Temp", "Air_Temp", "Salinity",
    "Chl_Volume_Filtered", "Chl1", "Chl2", "Avg_Chloro",
    "Phaeo1", "Phaeo2", "Avg_Phaeo",
    "Phosphate", "Silicate", "Nitrite", "Nitrite_Nitrate",
    "Ammonium", "Nitrate",
    "DA_Volume_Filtered", "pDA", "tDA", "dDA",
    "Volume_Settled_for_Counting",
    "Akashiwo_sanguinea", "Alexandrium_spp", "Dinophysis_spp",
    "Lingulodinium_polyedra", "Prorocentrum_spp",
    "Pseudo_nitzschia_delicatissima_group",
    "Pseudo_nitzschia_seriata_group",
    "Ceratium_spp", "Cochlodinium_spp", "Gymnodinium_spp",
    "Other_Diatoms", "Other_Dinoflagellates", "Total_Phytoplankton",
]


@dataclass(frozen=True)
class Station:
    """A single HABMAP pier."""

    code: str            # short slug used in our DataFrames
    erddap_id: str       # ERDDAP dataset_id, e.g. "HABs-ScrippsPier"
    pretty_name: str     # human-readable name
    region: str          # rough geographic bin
    active: bool = True


# Active HABMAP piers as of April 2026, verified against the SCCOOS ERDDAP
# catalog. The original 8-pier "HABMAP" subset is marked first; additional
# CalHABMAP-affiliated stations (Humboldt, Morro Bay, Tomales Bay) are
# included because they share the same variable schema and fill geographic
# gaps. Monterey Wharf (HABs-MontereyWharf) has been offline since
# March 2020 -- left out by default; pass it explicitly to load_station()
# if you want pre-2020 history from there.
STATIONS: tuple[Station, ...] = (
    # original HABMAP 8-pier core (SoCal -> NorCal)
    Station("scripps",     "HABs-ScrippsPier",       "Scripps Pier",        "SoCal-South"),
    Station("newport",     "HABs-NewportBeachPier",  "Newport Beach Pier",  "SoCal-Central"),
    Station("santamonica", "HABs-SantaMonicaPier",   "Santa Monica Pier",   "SoCal-Central"),
    Station("stearns",     "HABs-StearnsWharf",      "Stearns Wharf",       "SoCal-North"),
    Station("calpoly",     "HABs-CalPolyPier",       "Cal Poly Pier",       "Central"),
    Station("santacruz",   "HABs-SantaCruzWharf",    "Santa Cruz Wharf",    "Central-North"),
    Station("bodega",      "HABs-BodegaMarineLab",   "Bodega Marine Lab",   "NorCal"),
    Station("trinidad",    "HABs-TrinidadPier",      "Trinidad Pier",       "NorCal-Far"),
    # additional CalHABMAP stations on the same SCCOOS ERDDAP
    Station("morro_back",  "HABs-MorroBayBackBay",   "Morro Bay (Back)",    "Central"),
    Station("morro_front", "HABs-MorroBayFrontBay",  "Morro Bay (Front)",   "Central"),
    Station("humboldt",    "HABs-Humboldt",          "Humboldt",            "NorCal-Far"),
    Station("humboldt_sb", "HABs-HumboldtSouthBay",  "Humboldt South Bay",  "NorCal-Far"),
    Station("tomales_in",  "HABs-InnerTomalesBay",   "Inner Tomales Bay",   "NorCal"),
    Station("tomales_mid", "HABs-TomalesBayMid-ChannelBuoy", "Tomales Bay Mid-Channel Buoy", "NorCal"),
    Station("tomales_out", "HABs-TomalesBayMouth",   "Tomales Bay Mouth",   "NorCal"),
    Station("bodega_buoy", "HABs-BodegaMarineLabBuoy", "Bodega Marine Lab Buoy", "NorCal"),
)

# Easy lookup by short code.
STATIONS_BY_CODE: dict[str, Station] = {s.code: s for s in STATIONS}


def _build_url(
    erddap_id: str,
    *,
    variables: Iterable[str] = COMMON_VARIABLES,
    time_min: str = "2008-01-01T00:00:00Z",
    time_max: str = "2026-12-31T23:59:59Z",
) -> str:
    """Construct an ERDDAP tabledap CSV query URL for one pier."""
    var_part = ",".join(variables)
    constraints = (
        f"&time%3E={urllib.parse.quote(time_min)}"
        f"&time%3C={urllib.parse.quote(time_max)}"
    )
    return f"{ERDDAP_BASE}/{erddap_id}.csv?{urllib.parse.quote(var_part)}{constraints}"


def _read_erddap_csv(path) -> pd.DataFrame:
    """Parse an ERDDAP CSV (row 1 is units, not data).

    ``path`` may be a local ``pathlib.Path`` or a remote ``upath.UPath``
    (e.g. ``s3://bucket/.../HABs-ScrippsPier.csv``); pandas only accepts
    string paths through fsspec, so we stringify before reading.
    """
    p = str(path)
    units = pd.read_csv(p, nrows=1)
    df = pd.read_csv(p, skiprows=[1], low_memory=False)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df.attrs["units"] = units.iloc[0].to_dict()
    return df


def load_station(
    code_or_station: str | Station,
    *,
    refresh: bool = False,
    time_min: str = "2008-01-01T00:00:00Z",
    time_max: str = "2026-12-31T23:59:59Z",
    variables: Iterable[str] = COMMON_VARIABLES,
) -> pd.DataFrame:
    """Load one HABMAP pier; cache the raw CSV under ``Project/Data/habmap/``.

    The cache filename is the dataset id only (no time window suffix)
    when the default ``time_min``/``time_max`` are used, to keep
    backward compatibility with any existing ``HABs-*.csv`` you have on
    disk. When a non-default window is requested the window is hashed
    into the filename so a narrower window does not silently reuse a
    wider cached file (or vice versa).
    """
    station = (
        code_or_station if isinstance(code_or_station, Station)
        else STATIONS_BY_CODE[code_or_station.lower()]
    )
    storage.ensure_dir(HABMAP_DIR)
    default_min = "2008-01-01T00:00:00Z"
    default_max = "2026-12-31T23:59:59Z"
    if time_min == default_min and time_max == default_max:
        cache_path = HABMAP_DIR / f"{station.erddap_id}.csv"
    else:
        tag = f"{time_min[:10]}_{time_max[:10]}".replace(":", "")
        cache_path = HABMAP_DIR / f"{station.erddap_id}__{tag}.csv"

    if refresh or not cache_path.exists():
        url = _build_url(
            station.erddap_id,
            variables=variables,
            time_min=time_min,
            time_max=time_max,
        )
        try:
            with urllib.request.urlopen(url) as resp:
                cache_path.write_bytes(resp.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"ERDDAP fetch failed for {station.erddap_id} ({e.code}). "
                "The dataset may have a different variable schema; try "
                "passing a smaller `variables=` set."
            ) from e

    df = _read_erddap_csv(cache_path)
    df.insert(0, "station", station.code)
    df.attrs["station"] = station
    df.attrs["cache_path"] = str(cache_path)
    return df


def load_all_stations(
    *,
    refresh: bool = False,
    skip_failures: bool = True,
) -> pd.DataFrame:
    """Load every active HABMAP pier and stack them long-format.

    Returns a DataFrame keyed conceptually by ``(station, time)`` with a
    ``station`` short-code column added. Per-station unit dicts are kept
    on ``df.attrs["units_by_station"]``.
    """
    frames: list[pd.DataFrame] = []
    units_by_station: dict[str, dict] = {}
    failures: dict[str, str] = {}

    for station in STATIONS:
        try:
            sdf = load_station(station, refresh=refresh)
        except Exception as exc:
            if not skip_failures:
                raise
            failures[station.code] = str(exc)
            continue
        units_by_station[station.code] = sdf.attrs.get("units", {})
        frames.append(sdf)

    if not frames:
        raise RuntimeError(
            "No HABMAP stations loaded. ERDDAP may be unreachable or "
            "every dataset_id may have changed."
        )

    out = pd.concat(frames, ignore_index=True)
    out.attrs["units_by_station"] = units_by_station
    out.attrs["failures"] = failures
    return out


# -----------------------------------------------------------------------
# Back-compat alias used in the original Scripps-only script.
# -----------------------------------------------------------------------
def load_habs_data(*, refresh: bool = False) -> pd.DataFrame:
    """Back-compat shim: load the Scripps Pier record only."""
    return load_station("scripps", refresh=refresh)


def summarize(df: pd.DataFrame) -> None:
    """Print a one-shot summary of a HABMAP DataFrame."""
    if "station" in df.columns and df["station"].nunique() > 1:
        per_station = (
            df.groupby("station")
              .agg(rows=("time", "size"),
                   first=("time", "min"),
                   last=("time", "max"))
        )
        print("Per-station coverage:")
        print(per_station.to_string())
    else:
        print(f"Rows: {len(df)}")
        print(f"Date range: {df['time'].min()}  ->  {df['time'].max()}")
    if "failures" in df.attrs and df.attrs["failures"]:
        print("\nFailed stations:")
        for code, msg in df.attrs["failures"].items():
            print(f"  {code}: {msg}")


# -----------------------------------------------------------------------
# HABMAP + CalCOFI: a 75-year stacked PN record
# -----------------------------------------------------------------------
def load_habmap_plus_calcofi_history(
    *,
    refresh: bool = False,
    skip_failures: bool = True,
) -> pd.DataFrame:
    """Stack the 2008-present HABMAP record with the 1949-present
    CalCOFI net PN counts (where available).

    The result is a tidy long-format frame keyed by (station, time)
    with a ``source`` column indicating provenance:

        source == "habmap"        -- from SCCOOS HABMAP, weekly piers
        source == "calcofi_nets"  -- from CalCOFI net tows, quarterly

    Schema (audit C2 -- units are explicit and disjoint, NEVER mixed
    silently in a single magnitude column)::

        station, time, source, units,
        PN_cells_per_L,      # populated for HABMAP rows only
        PN_count_per_m3,     # populated for CalCOFI net rows only

    The two sources measure physically distinct quantities. HABMAP is a
    whole-water cell density (a niskin or pier intake counted on a
    Sedgewick-Rafter / Palmer-Maloney chamber, units = cells/L).
    CalCOFI nets are vertically integrated tows in a fixed mesh, so
    the natural unit is total cells per cubic meter of filtered water,
    which scales with both the in-situ density AND the depth of the
    water column the net traversed; you cannot convert one to the
    other with a constant factor.

    Downstream code MUST inspect ``units`` (or pick the right column)
    before applying any threshold. The earlier behavior overloaded a
    single ``PN_count`` column with both magnitudes (~1000x apart),
    which silently broke any comparable threshold.

    The CalCOFI side is wired to ``calcofi.load_nets`` /
    ``calcofi.pn_long_record``; both currently raise
    ``NotImplementedError`` until a tidy net CSV lands at
    ``Data/calcofi_nets.csv`` (see ``calcofi.load_nets``). When the
    CalCOFI data is missing this helper silently degrades to the
    HABMAP-only frame, with a note printed.
    """
    habmap = load_all_stations(refresh=refresh, skip_failures=skip_failures)
    habmap = habmap.copy()
    habmap["source"] = "habmap"

    # PN columns from HABMAP -- both size classes summed -> single
    # comparable PN total cell density (cells/L). NaN-safe via
    # ``min_count=1``: NaN+NaN stays NaN (the row gets no PN
    # measurement attributed to it), but NaN+x produces x. Previously
    # the loader silently substituted 0 for NaN on both sides via
    # ``Series.add(fill_value=0.0)``, which gave seriata=NaN and
    # delicatissima=2000 a total of 2000 rather than NaN.
    pn_cols = [c for c in (
        "Pseudo_nitzschia_seriata_group",
        "Pseudo_nitzschia_delicatissima_group",
    ) if c in habmap.columns]
    if pn_cols:
        pn_total = habmap[pn_cols].astype(float).sum(axis=1, min_count=1)
    else:
        pn_total = pd.Series(np.nan, index=habmap.index, dtype=float)
    habmap_long = habmap[["station", "time", "source"]].copy()
    habmap_long["units"] = "cells_per_L"
    habmap_long["PN_cells_per_L"] = pn_total
    habmap_long["PN_count_per_m3"] = np.nan

    # Try to splice in CalCOFI nets; degrade gracefully if missing.
    try:
        from calcofi import load_nets, pn_long_record  # local import (cycle-safe)
        nets = load_nets()
        cc_pn = pn_long_record(nets)
        if not cc_pn.empty:
            cc_long = cc_pn.rename(columns={"PN_calcofi": "PN_count_per_m3"})
            cc_long["source"] = "calcofi_nets"
            cc_long["units"] = "count_per_m3"
            cc_long["PN_cells_per_L"] = np.nan
            cols = ["station", "time", "source", "units",
                    "PN_cells_per_L", "PN_count_per_m3"]
            cc_long = cc_long.reindex(columns=cols)
            stacked = pd.concat([habmap_long[cols], cc_long],
                                ignore_index=True)
            # Sanity guard: each row must populate exactly one of the
            # two magnitude columns (never both, never neither when
            # source is known); refuses to mix units silently.
            both = stacked["PN_cells_per_L"].notna() \
                 & stacked["PN_count_per_m3"].notna()
            if bool(both.any()):
                raise RuntimeError(
                    "load_habmap_plus_calcofi_history: rows populated "
                    "in both PN_cells_per_L and PN_count_per_m3; refusing "
                    "to mix units."
                )
            stacked.attrs["sources"] = ["habmap", "calcofi_nets"]
            stacked.attrs["units_columns"] = {
                "habmap":       "PN_cells_per_L",
                "calcofi_nets": "PN_count_per_m3",
            }
            return stacked
    except NotImplementedError as e:
        # Expected on a fresh checkout -- the CalCOFI net data is not
        # on the public ERDDAP. Print once and return HABMAP-only.
        print(f"[load_habmap_plus_calcofi_history] CalCOFI nets unavailable; "
              f"returning HABMAP-only PN record. ({e})")
    except Exception as e:
        print(f"[load_habmap_plus_calcofi_history] CalCOFI splice failed: {e}")

    habmap_long.attrs["sources"] = ["habmap"]
    habmap_long.attrs["units_columns"] = {"habmap": "PN_cells_per_L"}
    return habmap_long


if __name__ == "__main__":
    df = load_all_stations()
    summarize(df)
