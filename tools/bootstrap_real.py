#!/usr/bin/env python3
"""Bootstrap 95% confidence intervals for the SIM-TO-REAL (real double-pendulum)
results.

Reads:
  data/real_validation/summary_multi.csv        (N=29 real clips; ESN/PINN/Hybrid horizons)
  data/real_validation/lam0_vs_lam1_real.csv     (PAIRED PINN horizons, physics loss on/off)

CLIP SCOPE (important): the canonical real set is 29 clips (IMG_1430..; clip 1392
was ruled invalid and dropped -- see data/canonical_results_B.md). summary_multi.csv
already holds exactly those 29. lam0_vs_lam1_real.csv was written BEFORE the drop
and still contains 30 rows including clip 1392, so Part B filters it down to the
same 29-clip scope before computing anything; the 30-row figure is printed
alongside for transparency only.

Outputs point estimates with 95% percentile bootstrap CIs and a paired Wilcoxon
test. Does NOT modify any .tex, any data file, or any existing script.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ---------------------------------------------------------------------------
# Reproducibility / bootstrap config
# ---------------------------------------------------------------------------
SEED = 12345
N_BOOT = 10000
LO_PCT, HI_PCT = 2.5, 97.5

np.random.seed(SEED)

# Resolve paths relative to this file's location so it runs from anywhere.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SUMMARY_CSV = os.path.join(ROOT, "data", "real_validation", "summary_multi.csv")
LAM_CSV = os.path.join(ROOT, "data", "real_validation", "lam0_vs_lam1_real.csv")


def percentile_ci(arr):
    """Return (lo, hi) 95% percentile interval of a bootstrap distribution."""
    return (float(np.percentile(arr, LO_PCT)), float(np.percentile(arr, HI_PCT)))


def boot_indices(n, n_boot):
    """Pre-generate all bootstrap row-index draws (n_boot x n)."""
    return np.random.randint(0, n, size=(n_boot, n))


# ===========================================================================
# Part A: summary_multi.csv -- per-model median horizons + PINN/ESN ratio
# ===========================================================================
summ = pd.read_csv(SUMMARY_CSV)
N_A = len(summ)

esn = summ["ESN_h"].to_numpy(dtype=float)
pinn = summ["PINN_h"].to_numpy(dtype=float)
hyb = summ["Hybrid_h"].to_numpy(dtype=float)

esn_cens = int(summ["ESN_cens"].sum())
pinn_cens = int(summ["PINN_cens"].sum())
hyb_cens = int(summ["Hybrid_cens"].sum())

# Per-clip PINN-over-ESN advantage ratio (real published ~2.8x).
per_clip_pinn_over_esn = pinn / esn

# Point estimates
esn_med = float(np.median(esn))
pinn_med = float(np.median(pinn))
hyb_med = float(np.median(hyb))
pinn_over_esn_ratio = float(np.median(per_clip_pinn_over_esn))

# Bootstrap (resample the 29 clips/rows with replacement; same row-draw applies
# to all quantities so it is a single consistent resample of the dataset).
idx_A = boot_indices(N_A, N_BOOT)

bs_esn_med = np.median(esn[idx_A], axis=1)
bs_pinn_med = np.median(pinn[idx_A], axis=1)
bs_hyb_med = np.median(hyb[idx_A], axis=1)
bs_pinn_over_esn = np.median(per_clip_pinn_over_esn[idx_A], axis=1)

esn_med_ci = percentile_ci(bs_esn_med)
pinn_med_ci = percentile_ci(bs_pinn_med)
hyb_med_ci = percentile_ci(bs_hyb_med)
pinn_over_esn_ci = percentile_ci(bs_pinn_over_esn)

# ===========================================================================
# Part B: lam0_vs_lam1_real.csv -- PAIRED physics-loss on vs off
# ===========================================================================
lam_raw = pd.read_csv(LAM_CSV)
N_B_RAW = len(lam_raw)

# Restrict to the canonical 29-clip scope defined by summary_multi.csv.
valid_tags = set(summ["tag"].tolist())
lam = lam_raw[lam_raw["tag"].isin(valid_tags)].reset_index(drop=True)
dropped_tags = sorted(set(lam_raw["tag"]) - valid_tags)
N_B = len(lam)

h1 = lam["h_lam1"].to_numpy(dtype=float)   # physics loss ON
h0 = lam["h_lam0"].to_numpy(dtype=float)   # physics loss OFF

lam1_cens = int(lam["cens_lam1"].sum())
lam0_cens = int(lam["cens_lam0"].sum())

per_clip_lam_ratio = h1 / h0

# Point estimates (two flavors of "advantage")
ratio_of_medians_pt = float(np.median(h1) / np.median(h0))
ratio_per_clip_pt = float(np.median(per_clip_lam_ratio))

# Bootstrap by resampling the PAIRED rows with replacement (keep pairs intact).
idx_B = boot_indices(N_B, N_BOOT)
h1_bs = h1[idx_B]
h0_bs = h0[idx_B]

bs_ratio_of_medians = np.median(h1_bs, axis=1) / np.median(h0_bs, axis=1)
bs_ratio_per_clip = np.median((h1_bs / h0_bs), axis=1)

ratio_of_medians_ci = percentile_ci(bs_ratio_of_medians)
ratio_per_clip_ci = percentile_ci(bs_ratio_per_clip)

# Paired Wilcoxon signed-rank test on the original (not bootstrapped) pairs.
# Differences that are exactly zero (identical horizons) are handled by scipy's
# default zero_method='wilcox' (drops zeros). Report p-value.
try:
    w_stat, w_p = wilcoxon(h1, h0)
    w_stat = float(w_stat)
    w_p = float(w_p)
except ValueError as e:
    w_stat, w_p = float("nan"), float("nan")
    print("Wilcoxon error:", e)

n_zero_diff = int(np.sum((h1 - h0) == 0))

# ===========================================================================
# Report
# ===========================================================================
def fmt_ci(ci):
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"

print("=" * 72)
print("SIM-TO-REAL bootstrap CIs  (seed=%d, n_boot=%d, pct=%.1f/%.1f)"
      % (SEED, N_BOOT, LO_PCT, HI_PCT))
print("=" * 72)

print("\n--- Part A: summary_multi.csv  (N=%d real clips) ---" % N_A)
print("Censored clips per model (cens=1, did NOT diverge in window):")
print("  ESN    : %d / %d" % (esn_cens, N_A))
print("  PINN   : %d / %d" % (pinn_cens, N_A))
print("  Hybrid : %d / %d" % (hyb_cens, N_A))
print()
print("Median horizon (seconds):")
print("  ESN_h    median = %.4f   95%% CI %s" % (esn_med, fmt_ci(esn_med_ci)))
print("  PINN_h   median = %.4f   95%% CI %s" % (pinn_med, fmt_ci(pinn_med_ci)))
print("  Hybrid_h median = %.4f   95%% CI %s" % (hyb_med, fmt_ci(hyb_med_ci)))
print()
print("PINN-over-ESN advantage  median(PINN_h/ESN_h)  [canonical B ratio-of-medians 2.50x]:")
print("  point = %.4f   95%% CI %s" % (pinn_over_esn_ratio, fmt_ci(pinn_over_esn_ci)))

print("\n--- Part B: lam0_vs_lam1_real.csv  (N=%d PAIRED clips) ---" % N_B)
print("Scope: file has %d rows; kept the canonical %d, dropped invalid clip(s) %s"
      % (N_B_RAW, N_B, dropped_tags if dropped_tags else "none"))
print("Censored: lam1 (physics ON) = %d / %d ; lam0 (physics OFF) = %d / %d"
      % (lam1_cens, N_B, lam0_cens, N_B))
print("Pairs with exactly zero horizon difference: %d / %d" % (n_zero_diff, N_B))
print()
print("lam1-vs-lam0 advantage  [canonical B per-run 1.16x; old 1.54x was point-mass]:")
print("  ratio of medians  median(h_lam1)/median(h_lam0)")
print("    point = %.4f   95%% CI %s" % (ratio_of_medians_pt, fmt_ci(ratio_of_medians_ci)))
print("  median per-clip ratio  median(h_lam1/h_lam0)")
print("    point = %.4f   95%% CI %s" % (ratio_per_clip_pt, fmt_ci(ratio_per_clip_ci)))
print()
print("Paired Wilcoxon signed-rank (h_lam1 vs h_lam0)  [NOT the paper's test, see NB below]:")
print("  statistic = %.4f   p-value = %.6g" % (w_stat, w_p))

# ===========================================================================
# Discrepancy honesty check vs published values
# ===========================================================================
print("\n--- Sanity vs data/canonical_results_B.md ---")
print("CAVEAT: the two CSVs read here are SINGLE-RUN. canonical_results_B.md reports")
print("the median over 8-10 independent retrains (data/canon_multirun/). Differences")
print("below are run-to-run scatter, not a contradiction -- quote canonical, not this.")
def near(x, target, tol_frac=0.10):
    return abs(x - target) <= tol_frac * target

pinn_over_esn_rom = pinn_med / esn_med
print("  PINN/ESN, canonical 2.50x (ratio-of-medians):")
print("    ratio-of-medians  = %.3f -> %s"
      % (pinn_over_esn_rom, "MATCH" if near(pinn_over_esn_rom, 2.50) else "DIFFERS"))
print("    per-clip ratio    = %.3f  (different statistic; not directly comparable)"
      % pinn_over_esn_ratio)
print("  physics-loss gain lam0.1/lam0, canonical 1.16x (0.417/0.358, per-run):")
print("    ratio-of-medians  = %.3f -> %s"
      % (ratio_of_medians_pt, "MATCH" if near(ratio_of_medians_pt, 1.16) else "DIFFERS"))
print("    per-clip ratio    = %.3f -> %s"
      % (ratio_per_clip_pt, "MATCH" if near(ratio_per_clip_pt, 1.16) else "DIFFERS"))
print("    (the older '1.54x' headline was point-mass calibre -- do NOT quote it)")
print("  Paired Wilcoxon lam1 vs lam0 : p=%.6g -> %s"
      % (w_p, "significant (<=0.01)" if (w_p == w_p and w_p <= 0.01) else "NOT significant"))
print("  NB: the paper's Wilcoxon (p=6e-4 / 4e-4) is PINN-vs-ESN and Hybrid-vs-ESN")
print("      on the 29 clips, NOT this lam1-vs-lam0 test.")

# Machine-readable dump for the orchestrator.
RESULTS = {
    "n_summary": N_A,
    "n_paired": N_B,
    "seed": SEED,
    "n_boot": N_BOOT,
    "censored_counts": {"ESN": esn_cens, "PINN": pinn_cens, "Hybrid": hyb_cens,
                         "lam1": lam1_cens, "lam0": lam0_cens},
    "real_esn_h_median": esn_med, "real_esn_h_ci": list(esn_med_ci),
    "real_pinn_h_median": pinn_med, "real_pinn_h_ci": list(pinn_med_ci),
    "real_hybrid_h_median": hyb_med, "real_hybrid_h_ci": list(hyb_med_ci),
    "real_pinn_over_esn_ratio": pinn_over_esn_ratio,
    "real_pinn_over_esn_ci": list(pinn_over_esn_ci),
    "lam1_over_lam0_ratio_of_medians": ratio_of_medians_pt,
    "lam1_over_lam0_ci": list(ratio_of_medians_ci),
    "lam1_over_lam0_ratio_per_clip": ratio_per_clip_pt,
    "lam1_over_lam0_per_clip_ci": list(ratio_per_clip_ci),
    "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
}
print("\nRESULTS_DICT =", RESULTS)
