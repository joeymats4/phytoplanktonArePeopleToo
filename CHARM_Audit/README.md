# CHARM_Audit -- self-contained C-HARM v3.1 fairness audit

This folder is a clean, self-contained snapshot of every file the C-HARM v3.1
fairness audit touches in the larger `Project/` codebase. It exists so a
teammate or reviewer can read, re-run, or hand off the audit without having to
navigate the rest of the project.

The headline claim being audited is:

> Our best LightGBM model (`lgb_arm_a_plus_calcofi`) beats the operational
> C-HARM v3.1 forecast on the HABMAP test fold (Nov 2022 -- 2025).

`CHARM_AUDIT.md` (the sidecar) is the canonical write-up. Read that first if
you only have time for one file.

## What lives here

```
CHARM_Audit/
├── README.md                     <- you are here
├── CHARM_AUDIT.md                <- the audit sidecar (canonical write-up)
├── requirements.txt              <- exact Python deps used in the run
│
├── charm.py                      <- audit-touching loader + helpers
│                                    (score_aggregator_panel,
│                                     fit_isotonic_temporal_split with
│                                     no-leakage assertion, 0.04 deg
│                                     box-at-pier sampler)
├── figures.py                    <- fig1 / fig1b / fig1c / fig1d
│                                    (headline + 3 supporting plots)
├── scripts/
│   ├── pull_charm_horizons.py    <- ERDDAP backfill for 0/2/3-day caches
│   └── charm_audit_panel.py      <- 4 fairness defenses + 5 sanity asserts
│
├── baselines.py                  <- imported: EVENT_TARGETS, splits
├── dataloading.py                <- imported: STATIONS, load_all_stations
├── evaluate.py                   <- imported: pr_auc, bootstrap_metric,
│                                              brier_decomposition,
│                                              reliability_table
├── replay.py                     <- imported by baselines.py (LATENCY,
│                                              cutoff, slice_available)
├── storage.py                    <- imported: parquet read/write helpers
│
├── Data/
│   ├── habmap/                   <- raw HABMAP station CSVs (16 piers)
│   ├── charm/wvcharmV3_{0,1,2,3}day/
│   │                             <- cached ERDDAP pulls per station/horizon
│   └── baselines/
│       ├── charm_aggregator_audit.parquet   (8 rows, drives fig1c)
│       ├── charm_horizon_audit.parquet      (8 rows, drives fig1d)
│       ├── charm_audit_summary.json         (machine-readable summary)
│       ├── predictions_p_pn_test.parquet    (LGB rows + 2 charm_* rows)
│       └── predictions_p_pda_test.parquet   (LGB rows + 2 charm_* rows)
│
└── plots/
    ├── fig1_pr_curves_vs_charm.png            (headline)
    ├── fig1b_reliability_vs_charm.png         (calibration story)
    ├── fig1c_charm_aggregator_robustness.png  (spatial-reduction defense)
    └── fig1d_charm_horizon_curve.png          (forecast-horizon defense)
```

## What's intentionally not here

- `run_baselines.py`, `gnn.py`, the CalCOFI / ERA5 / GLORYS / PACE / VIIRS
  feature builders, and the val/extreme-2015 prediction parquets. The audit
  consumes the **already-written** `predictions_p_*_test.parquet` model rows
  (LightGBM, climatology, persistence) for `lgb_arm_a_plus_calcofi`; it does
  not retrain anything, so none of the upstream training code is needed to
  reproduce the audit numbers.
- `fig2`–`fig5`. They are produced by the same `figures.py` module but read
  from artifacts unrelated to the C-HARM defense; only the `fig1` family is
  audit-relevant.

## How to reproduce from this folder

```bash
# from inside CHARM_Audit/
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (1) optional: re-pull the 0/2/3-day ERDDAP caches (skip if Data/charm/ is
#     populated, which it already is in this snapshot)
python scripts/pull_charm_horizons.py

# (2) re-score under all four fairness defenses (~5 min, CPU-only)
python scripts/charm_audit_panel.py

# (3) render the four fig1 plots
python figures.py
```

Steps (2) and (3) are idempotent: re-running (2) overwrites the audit
parquets and the appended `charm_*` rows in the predictions parquets but
does not touch any other model rows. Re-running (3) overwrites the four
PNGs in `plots/`.

## Headline result (row-aligned test slice)

| Target | n_aligned | base rate | Our best (lgb_arm_a + CalCOFI) | C-HARM raw 1-day | C-HARM post-isotonic | climatology | persistence |
|---|---|---|---|---|---|---|---|
| `p_pn`  | 865 | 0.275 | **0.501 [0.41, 0.57]** | 0.307 [0.22, 0.38] | 0.298 [0.22, 0.38] | 0.386 [0.31, 0.50] | 0.275 [0.20, 0.36] |
| `p_pda` | 840 | 0.037 | **0.129 [0.08, 0.20]** | 0.032 [0.03, 0.05] | 0.045 [0.03, 0.06] | 0.058 [0.04, 0.11] | 0.037 [0.03, 0.04] |

ΔPR-AUC vs raw C-HARM: **+0.19 on `p_pn`**, **+0.10 on `p_pda`**, with 95%
station-block bootstrap CIs that exclude zero on both targets. See
`CHARM_AUDIT.md` for the per-defense breakdown and the five sanity
assertions, all of which pass.
