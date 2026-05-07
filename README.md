# FinDS: The Tin Inversion and Frequency-Band Granger Feature Selection

Official code, data, and reproducibility artifacts for the paper:

> **The Tin Inversion: When Feature Importance Depends on Forecast Horizon in Multi-Horizon Time Series Forecasting**
> *Proceedings of the Workshop on Data Management for Modern Financial Systems (FinDS '26), Bengaluru, India.*

This repository accompanies the paper and provides everything needed to reproduce the 465-configuration Diebold–Mariano (DM) ablation study, the thirteen static feature-selection baselines, and the proposed **Frequency-Band Granger Feature Selection (FBGFS)** method on London Metal Exchange (LME) base metals data.

---

## TL;DR

Static feature-selection methods (Pearson, Spearman, mutual information, Granger causality, wavelet coherence, transfer entropy, and ten others) all rank **Tin (Sn)** last as a predictor of Aluminium prices. Yet a full combinatorial DM ablation over a Residual-GCN trained on LME daily prices (2008–2026, *n* = 4,569 days) shows that removing Tin produces the **single largest forecast degradation** at the longest horizon (DM = −27.6, *p* < 0.001 at *h* = 63 days, ℓ = 10).

We call this the **Tin Inversion**. The paper introduces **FBGFS**, a pre-model Granger-inspired method that produces a three-dimensional importance tensor Φ(*f*, *h*, ℓ) and recovers the inversion from raw price data alone, achieving **73% agreement** with the DM ground truth versus **20% for all thirteen static methods** — a 53 percentage-point improvement, computed in under 30 seconds on a single CPU core.

---

## Repository Structure

```
FinDS/
│
├── 1. Main Exp/                  # Stage 1: Res-GCN ablation study (Sections 4 & 6 of the paper)
│   └── Trains 31 × 5 × 3 = 465 Residual-GCNs and computes signed DM statistics
│       with HLN small-sample correction and Newey–West HAC variance.
│
├── 2. Static_Features_selection/ # Stage 2: Thirteen static baselines (Section 5)
│   └── Pearson, Spearman, Kendall τ, dCor, HSIC, MI, Transfer Entropy,
│       Conditional TE, Tail Dep., Wavelet (D5), Spectral Coherence,
│       Lagged Partial Corr., Granger, Frequency-Domain Granger.
│
├── 3. FBGFS_Analysis/            # Stage 3: FBGFS implementation and validation (Sections 7 & 8)
│   └── Builds the IPR² tensor Φ(f, h, ℓ) via horizon-scaled rolling features
│       and purged walk-forward CV with Ridge surrogate. Validates against
│       the DM ground truth produced in Stage 1.
│
├── Metals_Price.csv              # LME daily settlement prices, 6 metals, Jan 2008 – Feb 2026
├── coherence.csv                 # Welch periodogram coherence (per frequency)
├── band_coherence.csv            # Mean squared coherence per frequency band (Figure 2)
├── summary.csv                   # Aggregated DM / MAPE results across 465 configurations
├── fig_spectral_coherence.png    # Figure 2: Tin's +118% coherence gradient
├── fig_dual_analysis.png         # Figure 1: DM trajectories across horizons
└── README.md
```

The folders are numbered in the order they should be run for a full replication. Each folder contains Jupyter notebooks; all code is notebook-based and self-contained.

---

## Dataset

`Metals_Price.csv` contains LME official daily settlement prices for six base metals over **January 2008 to February 2026** (*n* = 4,569 trading days):

| Symbol | Metal | Role |
|--------|-------|------|
| Al | Aluminium | Forecast target |
| Cu | Copper | Auxiliary feature |
| Ni | Nickel | Auxiliary feature |
| Zn | Zinc | Auxiliary feature |
| Sn | Tin | Auxiliary feature *(the inversion)* |
| Pb | Lead | Auxiliary feature |

All models operate on log-returns. The chronological train/val/test split is **70 / 15 / 15** with a fixed random seed of 123. Lookback windows ℓ ∈ {10, 22, 63} correspond to two weeks, one month, and one quarter; forecast horizons *h* ∈ {1, 5, 10, 22, 63} span daily to quarterly.

---

## Experimental Pipeline

### Stage 1 — Residual-GCN ablation (`1. Main Exp/`)

A three-layer Residual-GCN with hidden dimension 32 and dropout 0.15 is trained independently for every combination of:

- 31 non-trivial subsets of the five auxiliary metals (2⁵ − 1)
- 5 forecast horizons *h*
- 3 lookback windows ℓ

producing **465 trained models**. For each ablation we compute the signed Diebold–Mariano statistic with the Harvey–Leybourne–Newbold small-sample correction and Newey–West HAC variance (Bartlett kernel, bandwidth = max(1, *h* − 1)). Outputs include MAE, RMSE, MAPE, and NRMSE; MAPE is the primary headline metric in the paper.

### Stage 2 — Static feature selection (`2. Static_Features_selection/`)

Thirteen pre-model criteria are computed on log-returns of all five auxiliary metals against Aluminium and Z-score normalised across features within each method:

Broadband linear: **Pearson, Spearman, Kendall τ**
Nonlinear dependence: **dCor, HSIC**
Information-theoretic: **Mutual Information, Transfer Entropy, Conditional Transfer Entropy**
Spectral: **Wavelet correlation (D5), Welch spectral coherence**
Granger-type: **Lagged partial correlation, bivariate Granger, Breitung–Candelon frequency-domain Granger**
Distributional: **Tail dependence**

Tin ranks last on 11 of 13 criteria with a mean Z-score of −1.232. None of these methods conditions on horizon or lookback.

### Stage 3 — FBGFS analysis (`3. FBGFS_Analysis/`)

Implements **Frequency-Band Granger Feature Selection**. For each candidate feature *f*, four horizon-scaled signals are constructed:

- φ₁ = √*h* · mean log-return over the lookback (horizon-aware drift)
- φ₂ = standard deviation of log-returns over the lookback
- φ₃ = lag-1 log-return (most recent direction)
- φ₄ = rolling Pearson correlation of *f* and the target over the lookback

Incremental Predictive R² is then estimated as

> Φ(*f*, *h*, ℓ) = R²_full(*h*, ℓ) − R²_{\\f}(*h*, ℓ)

using **purged walk-forward cross-validation** (5 folds, purge gap = *h*) with a **Ridge surrogate** (regularisation α selected by inner 3-fold CV over 15 logarithmically spaced values in {10⁻³, …, 10³}). Feature removal is implemented by zeroing the four signal columns rather than dropping them, preserving dimensional consistency across folds.

The output is the three-dimensional tensor Φ ∈ ℝ^(|F|×|H|×|L|), computed in **under 30 seconds on a single CPU core** versus approximately **15 GPU-hours** for the 465-model GCN ablation.

---

## Headline Results

### Tin Inversion (Section 6.2)

| Lookback ℓ | h = 1 | h = 5 | h = 10 | h = 22 | h = 63 |
|------------|-------|-------|--------|--------|--------|
| 10 | −2.58 | −7.54 | −2.62 | −6.39 | **−27.64** |
| 22 | +6.22 | −3.21 | −7.41 | −7.54 | **−20.65** |
| 63 | −5.47 | +9.80 | +1.37 | +3.47 | **−15.48** |

Signed DM for single-Tin ablation. Negative values mean the full model beats the ablated model (Tin matters). Bold = ℎ = 63 column. Tin flips from harmful at short horizons to dominant at long horizons.

### FBGFS vs. static methods (Section 8)

| Selection method | DM-agreement on Tin top-3 |
|------------------|---------------------------|
| All 13 static methods (Pearson, Granger, MI, …) | **3 / 15 (20%)** |
| **FBGFS (this work)** | **11 / 15 (73%)** |

FBGFS recommends including Tin in 14 of 15 (*h*, ℓ) configurations; the static methods exclude it from all 15.

### Spectral mechanism (Section 6.4)

Tin's mean squared coherence with Aluminium rises from Ĉ = 0.198 at < 5-day cycles to Ĉ = 0.434 at > 63-day cycles — a **+118% gradient**, the steepest among all five metals. Broadband association measures average over a band where Tin is weakest, which is why all 13 static methods miss it. See `fig_spectral_coherence.png`.

---

## Reproducing the Paper

1. Clone the repository and install dependencies (PyTorch, NumPy, Pandas, scikit-learn, statsmodels, PyWavelets, scipy).
2. Run notebooks in order, top to bottom, within each numbered folder:
   - `1. Main Exp/` produces the 465-model DM ablation. Plan for ~15 GPU-hours on a single GPU.
   - `2. Static_Features_selection/` produces Table 1 (Z-score ranking, ~minutes on CPU).
   - `3. FBGFS_Analysis/` produces Tables 4 and 5 and the 73% agreement result (~30 seconds on CPU).
3. Aggregate CSVs (`summary.csv`, `coherence.csv`, `band_coherence.csv`) and figures (`fig_*.png`) are committed at the repo root for inspection without re-running the GCN ablation.

Random seed 123 is fixed throughout; results should match the paper to within numerical tolerance.

---

## Evaluation Protocol

- **Diebold–Mariano** test under squared-error loss with HLN small-sample correction and Newey–West HAC variance (Bartlett kernel, bandwidth = max(1, *h* − 1)). All 465 tests reported at α = 0.05; 397 (85.4%) significant at α = 0.05, 370 (79.6%) significant at α = 0.01.
- **Forecast metrics:** MAE, RMSE, MAPE (primary), NRMSE.
- **Cross-validation:** purged walk-forward with purge gap = *h*, *K* = 5 outer folds, inner 3-fold CV for Ridge α selection inside FBGFS.

---

## When to Use FBGFS

The paper argues — and the data supports — that any multi-horizon financial forecasting pipeline performing static feature selection should consider replacing its global importance vector with a horizon-stratified importance tensor before model training. FBGFS is most useful when:

1. The forecast horizon spans more than one timescale (e.g. daily and quarterly together).
2. Some candidate features are suspected of carrying low-frequency signal (commodity supply cycles, macro indicators, seasonal agricultural drivers).
3. The downstream model is expensive to train, so an exponential search over feature subsets is infeasible.

The same machinery is expected to generalise to energy markets, agricultural commodities, and macroeconomic forecasting wherever predictive signal is concentrated in specific frequency bands.

---

## Limitations

- **Linear surrogate gap.** FBGFS uses Ridge regression as its surrogate. The four DM disagreements in Section 8 are likely attributable to non-linear GCN interaction effects that a linear model cannot replicate. Gradient-boosted or shallow-MLP surrogates are the natural next step.
- **Single target.** The ablation predicts Aluminium only; symmetry across all six metals requires per-target replication.
- **Static graph topology.** Edge weights are static Pearson correlations on the training fold. Replacing them with horizon-conditioned IPR² scores is on the roadmap.
- **Time-averaged ground truth.** DM statistics aggregate 2008–2026; regime-conditioned Φ tensors are needed to track Tin's importance across commodity super-cycles.

---

## Citation

If you use this code, data, or the FBGFS method, please cite the paper:

```bibtex
@inproceedings{TinInversion2026,
  title     = {The Tin Inversion: When Feature Importance Depends on Forecast Horizon in Multi-Horizon Time Series Forecasting},
  author    = {Muktinath Vishwakarma, Manish Kurhekar},
  booktitle = {Proceedings of the Workshop on Data Management for Modern Financial Systems (FinDS '26)},
  year      = {2026},
  address   = {Bengaluru, India},
  publisher = {ACM}
}
```

(Final author list and DOI will be added on acceptance.)

---

## License

Released under the MIT License. The LME settlement-price data in `Metals_Price.csv` is included for academic reproducibility only; downstream users should consult LME's data-licensing terms for any commercial use.

---

## Author

**Muktinath V** — AI Researcher, time-series forecasting and graph neural networks.
Repository: <https://github.com/prof-manav/FinDS>
