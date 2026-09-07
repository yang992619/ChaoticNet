#!/usr/bin/env python3
"""
Bootstrap 95% confidence intervals for the SIMULATION headline comparison.

*** CALIBRE / 口径: DISTRIBUTED-MASS (B version). ***
All numbers produced here are on the same footing as the single source of truth
data/canonical_results_B.md.  Do NOT substitute other horizon CSVs.

HISTORY / WARNING
-----------------
Until 2026-09 this script read data/v2_50trial_2026-06-11/horizon_compare_50.csv
as its "two-way canonical" source.  That file is a POINT-MASS-era artefact
(ESN median 0.150 s, PINN median 0.642 s) and is incompatible with the current
distributed-mass results (ESN 0.40 s, PINN 0.88 s).  It is archival only and is
NO LONGER READ by this script.  Never quote CIs derived from it.

Data sources (both distributed-mass):
  A) Per-run canonical (THE headline statistic in canonical_results_B.md):
        data/canon_multirun/insim_runs.csv   columns: model,run,trial,horizon
        data/canon_multirun/real_runs.csv    columns: model,run,tag,horizon
     Statistic: for each run take the median horizon over that run's CHAOTIC
     trials (horizon < 9.9 s) in-sim, or over all 29 real clips; then report the
     median across runs.  Bootstrap resamples RUNS with replacement
     (ESN n=10 runs, PINN/PINN0/Hybrid n=8 runs), so CIs are coarse by design --
     the run count, not the trial count, is the limiting sample size.
  B) Fixed-subset four-way cross-check ("四方固定子集交叉验证"):
        data/fair_compare/four_way_horizons.csv
        columns: trial,esn,pinn,hybrid,lam0
     Filter esn<9.9 AND pinn<9.9 AND hybrid<9.9 (N=32; the lam0 censored trial
     is inside the same 32, so the four-way subset is identical).  Bootstrap is
     PAIRED-ROW: resample trial indices with replacement, recompute on the
     resampled rows, so ratios stay per-trial paired.

Method: 10000 bootstrap resamples, percentile CI (2.5, 97.5).
Point estimates are computed directly from the data, not from the resamples.
"""

import os
import csv
import numpy as np

SEED = 12345
N_BOOT = 10000
LO_PCT, HI_PCT = 2.5, 97.5
CENSOR = 9.9  # horizons >= this hit the integration window and are non-chaotic

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INSIM_RUNS_CSV = os.path.join(ROOT, "data", "canon_multirun", "insim_runs.csv")
REAL_RUNS_CSV = os.path.join(ROOT, "data", "canon_multirun", "real_runs.csv")
FOURWAY_CSV = os.path.join(ROOT, "data", "fair_compare", "four_way_horizons.csv")

# Deliberately NOT a path constant any more -- kept only as a documented
# tombstone so nobody re-adds it by accident:
#   data/v2_50trial_2026-06-11/horizon_compare_50.csv  == POINT-MASS, DO NOT USE.

MODELS = ["ESN", "PINN", "PINN0", "Hybrid"]


def load_columns(path, cols):
    """Load named float columns from a CSV into a dict of numpy arrays."""
    data = {c: [] for c in cols}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for c in cols:
                data[c].append(float(row[c]))
    return {c: np.asarray(v, dtype=float) for c, v in data.items()}


def load_run_medians(path, chaos_only):
    """model -> (run_ids, per-run median horizon) for the given runs CSV."""
    buckets = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            h = float(row["horizon"])
            if chaos_only and h >= CENSOR:
                continue
            buckets.setdefault(row["model"], {}).setdefault(row["run"], []).append(h)
    out = {}
    for model, runs in buckets.items():
        run_ids = sorted(runs, key=lambda r: int(r))
        out[model] = (run_ids,
                      np.array([np.median(runs[r]) for r in run_ids], dtype=float))
    return out


def percentile_ci(vals):
    return (float(np.percentile(vals, LO_PCT)), float(np.percentile(vals, HI_PCT)))


def boot_ci(stat_fn, n_rows, rng):
    """Paired-row bootstrap: resample row indices, recompute stat, return (lo, hi)."""
    vals = np.empty(N_BOOT, dtype=float)
    for b in range(N_BOOT):
        vals[b] = stat_fn(rng.integers(0, n_rows, size=n_rows))
    return percentile_ci(vals)


def fmt(name, point, ci):
    print(f"  {name:30s} point = {point:.4f}   95% CI = [{ci[0]:.4f}, {ci[1]:.4f}]")


# ---------------------------------------------------------------------------
# Section A: per-run canonical bootstrap
# ---------------------------------------------------------------------------
def section_per_run(title, path, chaos_only, rng, results, prefix):
    med = load_run_medians(path, chaos_only)
    print("=" * 74)
    print(title)
    print(f"  source: {path}   (bootstrap unit = RUN)")

    # Pre-draw one bootstrap matrix of per-run medians per model.
    draws = {}
    for m in MODELS:
        run_ids, vals = med[m]
        n = len(vals)
        idx = rng.integers(0, n, size=(N_BOOT, n))
        draws[m] = np.median(vals[idx], axis=1)
        point = float(np.median(vals))
        ci = percentile_ci(draws[m])
        fmt(f"{m} median (s), n_runs={n}", point, ci)
        results[f"{prefix}_{m}_median"] = point
        results[f"{prefix}_{m}_ci"] = list(ci)
        results[f"{prefix}_{m}_n_runs"] = n

    # Ratios vs ESN. Runs of different models are INDEPENDENT retrains, not
    # paired, so the resamples are drawn independently -- report as such.
    for m in MODELS[1:]:
        point = float(np.median(med[m][1]) / np.median(med["ESN"][1]))
        ci = percentile_ci(draws[m] / draws["ESN"])
        fmt(f"{m}/ESN ratio (unpaired)", point, ci)
        results[f"{prefix}_{m}_over_ESN_ratio"] = point
        results[f"{prefix}_{m}_over_ESN_ci"] = list(ci)


# ---------------------------------------------------------------------------
# Section B: fixed-subset four-way, paired-row bootstrap
# ---------------------------------------------------------------------------
def section_fixed_subset(rng, results):
    cols = ["esn", "pinn", "hybrid", "lam0"]
    d = load_columns(FOURWAY_CSV, cols)
    mask = (d["esn"] < CENSOR) & (d["pinn"] < CENSOR) & (d["hybrid"] < CENSOR)
    sub = {c: d[c][mask] for c in cols}
    n = int(mask.sum())
    results["fixed_subset_N"] = n

    print("=" * 74)
    print(f"FIXED-SUBSET four-way chaos cross-check  N = {n}  (expected 32)")
    print(f"  source: {FOURWAY_CSV}   (bootstrap unit = TRIAL, paired rows)")
    if n != 32:
        print(f"  !! WARNING: N={n} != 32, data may have changed; check canonical_results_B.md")

    label = {"esn": "ESN", "pinn": "PINN", "hybrid": "Hybrid", "lam0": "PINN0 (lam=0 MLP)"}
    for c in cols:
        point = float(np.median(sub[c]))
        ci = boot_ci(lambda idx, c=c: np.median(sub[c][idx]), n, rng)
        fmt(f"{label[c]} median (s)", point, ci)
        results[f"fixed_{c}_median"] = point
        results[f"fixed_{c}_ci"] = list(ci)

    # NOTE two different "ratio" statistics -- do not mix them up:
    #   ratio-of-medians  = median(X)/median(ESN)  <- this is what
    #                       canonical_results_B.md quotes (1.72x / 3.0x / 7.38x)
    #   per-trial ratio   = median(X_i / ESN_i)    <- paired, systematically
    #                       smaller here because ESN's small horizons dominate.
    for c in ["pinn", "hybrid", "lam0"]:
        point = float(np.median(sub[c]) / np.median(sub["esn"]))
        ci = boot_ci(lambda idx, c=c: np.median(sub[c][idx]) / np.median(sub["esn"][idx]),
                     n, rng)
        fmt(f"{label[c]}/ESN ratio-of-medians", point, ci)
        results[f"fixed_{c}_over_esn_ratio_of_medians"] = point
        results[f"fixed_{c}_over_esn_ratio_of_medians_ci"] = list(ci)

        point_p = float(np.median(sub[c] / sub["esn"]))
        ci_p = boot_ci(lambda idx, c=c: np.median(sub[c][idx] / sub["esn"][idx]), n, rng)
        fmt(f"{label[c]}/ESN per-trial ratio", point_p, ci_p)
        results[f"fixed_{c}_over_esn_ratio"] = point_p
        results[f"fixed_{c}_over_esn_ci"] = list(ci_p)


def main():
    results = {"seed": SEED, "n_boot": N_BOOT, "calibre": "distributed_mass_B"}
    rng = np.random.default_rng(SEED)

    section_per_run("IN-SIM per-run canonical (chaotic trials, horizon < 9.9 s)",
                    INSIM_RUNS_CSV, True, rng, results, "insim")
    section_per_run("REAL zero-shot per-run canonical (29 clips, no censoring filter)",
                    REAL_RUNS_CSV, False, rng, results, "real")
    section_fixed_subset(rng, results)

    print("=" * 74)
    print("Compare against data/canonical_results_B.md (single source of truth).")
    return results


if __name__ == "__main__":
    main()
