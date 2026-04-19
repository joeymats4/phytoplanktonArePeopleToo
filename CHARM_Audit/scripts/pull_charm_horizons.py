"""Backfill the missing C-HARM v3.1 horizons for the fairness audit.

The headline run only used ``wvcharmV3_1day`` (1-day forecast) at all 16
HABMAP piers. To prove our beat-the-incumbent claim is invariant to the
forecast-horizon choice, we also need:

    * wvcharmV3_0day  -- nowcast: 13 piers missing locally (only newport,
                         santamonica, scripps were pulled originally)
    * wvcharmV3_2day  -- 2-day forecast: all 16 piers missing
    * wvcharmV3_3day  -- 3-day forecast: all 16 piers missing

This script is a thin orchestrator on top of charm.fetch_charm_all_stations
that (a) targets only the missing horizons, (b) is fully idempotent (skips
station/variable pairs whose cache file already exists), and (c) supports
``--dry-run`` so you can see the planned ERDDAP traffic before kicking it
off.

Wall-clock estimate: 30-60 min on the live CoastWatch ERDDAP, depending
on server load. Cache lives under
``Project/Data/charm/<dataset_id>/<station_code>/<var>__<start>_<stop>.nc``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from charm import (  # noqa: E402  (sys.path tweak above)
    DATA_DIR,
    VARIABLES,
    fetch_charm_at_station,
)
from dataloading import STATIONS  # noqa: E402


# Horizons we want filled in. wvcharmV3_1day is intentionally absent --
# the original ``scripts/pull_charm.py`` already cached it for all 16
# piers and we don't want to pay for a refresh by accident.
HORIZONS_TO_FILL: tuple[str, ...] = (
    "wvcharmV3_0day",
    "wvcharmV3_2day",
    "wvcharmV3_3day",
)


def _is_cached(
    dataset_id: str,
    station_code: str,
    variable: str,
    time_min: str,
    time_max: str,
) -> bool:
    """Match the cache-filename convention from ``charm.fetch_charm_at_station``.

    The cache key is ``<var>__<YYYY-MM-DD>_<YYYY-MM-DD>.nc`` where the
    dates are the (clamped) time window. We can't replicate the
    .das-driven clamping here without a network call, so we accept any
    cache file that starts with ``<var>__`` and assume that if the user
    is re-running this script with different time bounds, they'd pass
    --refresh.
    """
    out_dir = DATA_DIR / "charm" / dataset_id / station_code
    if not out_dir.exists():
        return False
    return any(out_dir.glob(f"{variable}__*.nc"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--time-min", default="2022-01-01T00:00:00Z")
    ap.add_argument("--time-max", default="2025-12-31T23:59:59Z")
    ap.add_argument(
        "--horizons", nargs="*", default=list(HORIZONS_TO_FILL),
        help="C-HARM dataset IDs to fill (default: 0day, 2day, 3day)",
    )
    ap.add_argument(
        "--variables", nargs="*", default=list(VARIABLES),
        help="variables to pull per dataset (default: the 3 event-prob vars)",
    )
    ap.add_argument(
        "--refresh", action="store_true",
        help="re-download even cached station/variable pairs",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="print the planned fetches and exit without touching the network",
    )
    args = ap.parse_args()

    plan: list[tuple[str, str]] = []  # (dataset_id, station_code)
    for ds_id in args.horizons:
        for s in STATIONS:
            need = args.refresh or not all(
                _is_cached(ds_id, s.code, v, args.time_min, args.time_max)
                for v in args.variables
            )
            if need:
                plan.append((ds_id, s.code))

    print(f"[pull_charm_horizons] horizons={args.horizons}")
    print(f"[pull_charm_horizons] variables={args.variables}")
    print(f"[pull_charm_horizons] window={args.time_min} .. {args.time_max}")
    print(f"[pull_charm_horizons] {len(plan)} (dataset, station) pairs to fetch:")
    for ds_id, code in plan:
        print(f"    {ds_id:18s}  {code}")

    if args.dry_run:
        print("[pull_charm_horizons] --dry-run; exiting without fetching")
        return
    if not plan:
        print("[pull_charm_horizons] nothing to do (everything cached)")
        return

    code_to_station = {s.code: s for s in STATIONS}
    t0 = time.time()
    successes = 0
    failures: list[tuple[str, str, str]] = []  # (ds_id, code, error)
    for i, (ds_id, code) in enumerate(plan, start=1):
        s = code_to_station[code]
        elapsed = time.time() - t0
        print(
            f"[{i:3d}/{len(plan)}] {ds_id:18s}  {code:14s}  "
            f"(elapsed {elapsed:5.1f}s)",
            flush=True,
        )
        try:
            fetch_charm_at_station(
                s,
                dataset_id=ds_id,
                variables=args.variables,
                time_min=args.time_min,
                time_max=args.time_max,
                refresh=args.refresh,
            )
            successes += 1
        except Exception as exc:
            print(f"    FAILED: {exc}")
            failures.append((ds_id, code, str(exc)))

    elapsed = time.time() - t0
    print(
        f"\n[pull_charm_horizons] done in {elapsed:.1f}s  "
        f"successes={successes}  failures={len(failures)}"
    )
    if failures:
        print("[pull_charm_horizons] failure summary:")
        for ds_id, code, err in failures:
            print(f"    {ds_id} / {code}: {err}")


if __name__ == "__main__":
    main()
