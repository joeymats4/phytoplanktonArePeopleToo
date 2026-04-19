"""Storage abstraction: local ``pathlib.Path`` or remote ``UPath``.

Every module that today hardcodes ``Path(__file__).resolve().parent / "Data"``
should instead read its data root through this module, so the same code
runs against either a local checkout or an S3-backed cache without
branching at the call site.

Switches based on environment variables:

    DH2026_DATA_ROOT   unset                  -> ``Project/Data/``         (local laptop)
                       ``s3://bucket/Data``   -> S3 via fsspec/s3fs        (EC2/SageMaker in us-west-2)
                       ``/some/other/path``   -> that local directory      (shared NFS, etc.)

    DH2026_LOCAL_SCRATCH                      -> override for the
                                                 always-local cache used
                                                 by tools that can only
                                                 write to a real filesystem
                                                 (cdsapi, copernicusmarine,
                                                 earthaccess.download).

Why two roots? ``pandas.read_parquet``, ``pandas.read_csv``, ``to_parquet``,
``to_csv``, and ``xarray.open_mfdataset`` all dispatch to fsspec when handed
an ``s3://...`` string, so the bulk of the codebase can target S3
transparently. But three of our pull tools (cdsapi for ERA5,
copernicusmarine for GLORYS, earthaccess.download for raw NASA netCDFs)
write to a real local file path internally and don't grow an fsspec back
end. Those modules should call ``local_scratch()`` instead of
``data_root()`` for their binary cache directories. When ``DH2026_DATA_ROOT``
points at S3 and ``DH2026_LOCAL_SCRATCH`` is unset, ``local_scratch()``
falls back to ``$TMPDIR/dh2026`` so the local copy is at least bounded.

Lazy import of ``upath`` keeps a fresh checkout importable without the
optional ``universal_pathlib`` / ``s3fs`` dependencies.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

_LOCAL_DEFAULT: Path = Path(__file__).resolve().parent / "Data"
_LOCAL_PLOTS_DEFAULT: Path = Path(__file__).resolve().parent / "plots"

# Canonical names of the per-source subdirectories under ``data_root()``.
# Importable so callers don't have to guess at strings.
DATASET_DIRS: tuple[str, ...] = (
    "habmap",     # SCCOOS HABMAP pier CSVs (HABs-*.csv)
    "climate",    # CUTI/BEUTI + ONI/PDO/MEI/NPGO indices
    "calcofi",    # CalCOFI bottle / pier-panel / climatology parquets
    "satellite",  # NASA Earthdata caches (PACE/MODIS/VIIRS) + assembled panels
    "charm",      # C-HARM v3.1 forecast cache
    "baselines",  # baseline_table.csv, features_smoke.parquet, etc.
)

# Schemes we treat as "remote" -- anything fsspec can route via a URL.
# Local mounts (``/mnt/foo``, ``./Data``) do not have ``://``.
_REMOTE_PREFIXES: tuple[str, ...] = (
    "s3://", "gs://", "az://", "abfs://", "abfss://", "https://", "http://",
)


def _is_remote_str(s: str) -> bool:
    return s.startswith(_REMOTE_PREFIXES)


def data_root() -> Any:
    """Return the project data root as a Path-like object.

    The return type is ``pathlib.Path`` for local roots and
    ``upath.UPath`` for remote roots; both expose ``.joinpath``,
    ``.exists``, ``.write_bytes``, ``.read_bytes``, ``.open``, ``.glob``,
    and stringify to either a regular path or an ``s3://...`` URL.
    """
    root = os.environ.get("DH2026_DATA_ROOT", "").strip()
    if not root:
        return _LOCAL_DEFAULT
    if _is_remote_str(root):
        from upath import UPath  # lazy: only required when actually using S3
        return UPath(root)
    return Path(root)


def local_scratch() -> Path:
    """Always-local Path for binary download caches.

    Use for any tool that writes to a real filesystem: cdsapi,
    copernicusmarine, earthaccess.download, or the streaming-to-tempfile
    pattern in ``calcofi.load_bottle``.

    Resolution order:
        1. ``$DH2026_LOCAL_SCRATCH`` if set,
        2. ``$TMPDIR/dh2026`` when ``DH2026_DATA_ROOT`` points at a remote URL,
        3. ``Project/Data/`` (the local default).
    """
    env = os.environ.get("DH2026_LOCAL_SCRATCH", "").strip()
    if env:
        p = Path(env)
        p.mkdir(parents=True, exist_ok=True)
        return p
    if _is_remote_str(os.environ.get("DH2026_DATA_ROOT", "")):
        p = Path(tempfile.gettempdir()) / "dh2026"
        p.mkdir(parents=True, exist_ok=True)
        return p
    return _LOCAL_DEFAULT


def cache_path(*parts: str) -> Any:
    """``data_root().joinpath(*parts)`` -- short alias for callers that
    don't want to import ``data_root`` and join manually."""
    return data_root().joinpath(*parts)


def dataset_dir(name: str) -> Any:
    """Return ``data_root() / <name>/`` and make sure it exists.

    Use this from every loader / puller so the ``Data/`` tree stays
    organized: HABMAP CSVs live under ``Data/habmap/``, climate-index
    files under ``Data/climate/``, parquets under ``Data/calcofi/`` or
    ``Data/satellite/`` etc. Names should be one of ``DATASET_DIRS``;
    arbitrary names are allowed but discouraged.
    """
    target = data_root().joinpath(name)
    ensure_dir(target)
    return target


def plots_root() -> Any:
    """Return the project plots root as a Path-like object.

    Mirrors :func:`data_root` but defaults to ``Project/plots/``. Set
    ``DH2026_PLOTS_ROOT`` to override (e.g. ``s3://bucket/plots`` or a
    notebooks scratch directory). Figures live under here, never under
    ``data_root()``, so the data cache is purely *data*.
    """
    root = os.environ.get("DH2026_PLOTS_ROOT", "").strip()
    if not root:
        return _LOCAL_PLOTS_DEFAULT
    if _is_remote_str(root):
        from upath import UPath
        return UPath(root)
    return Path(root)


def ensure_dir(p: Any) -> None:
    """``mkdir(parents=True, exist_ok=True)`` that's a no-op on object
    stores (S3 has no real directories; some UPath backends raise)."""
    try:
        p.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass
    except (NotImplementedError, OSError):
        # S3 / GCS: directory creation is meaningless. Leave the
        # ``s3://bucket/prefix`` namespace untouched -- the first object
        # write will materialize the prefix.
        pass


def is_remote(p: Any = None) -> bool:
    """True if ``p`` (or ``data_root()`` when ``p`` is None) lives on a
    remote object store rather than a local filesystem."""
    target = data_root() if p is None else p
    return _is_remote_str(str(target))


def fspath(p: Any) -> str:
    """Stringify a Path or UPath in the form pandas/xarray prefer.

    For local paths returns ``str(p)``; for UPath returns the canonical
    ``s3://bucket/key`` URL (which fsspec then routes correctly).
    """
    s = str(p)
    return s
