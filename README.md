# MoE-CP reproduction

Reimplementation of **Adaptive Conformal Prediction via Mixture-of-Experts Gating Similarity**
(Kong et al., ICLR 2026) and of the paper's baselines CQR, CHR, SCP+CC, RLCP and PCP.
Not included: the appendix ablations.

| Method | Reference | Base model (paper App. A.4.1) | Code |
|---|---|---|---|
| CQR | Romano et al. 2019 | quantile RF: 100 trees, min leaf 50, quantiles 0.01…0.99 | `moecp/baselines/cqr.py` |
| CHR | Sesia & Romano 2021 | same quantile RF | `moecp/baselines/chr.py` |
| SCP+CC | Gibbs et al. 2025 | `RandomForestRegressor()`, default settings | `moecp/baselines/cc.py` |
| RLCP | Hore & Barber 2025 | `RandomForestRegressor()`, default settings | `moecp/baselines/rlcp.py` |
| PCP | Zhang & Candès 2024 | `RandomForestRegressor()`, default settings | `moecp/baselines/pcp.py` |
| MoE-CP | this paper | MoE (K experts + gate) | `moecp/conformal.py` |

The baselines are reimplemented from their papers and the authors' public code
(msesia/chr, yaozhang24/pcp). That code has no license, so none of it is copied.

## Layout
| Path | Contents |
|---|---|
| `moecp/moe.py` | MoE regressor: K MLP experts + MLP gate, `mu = Σ π_k μ_k` (Eqs. 1–2) |
| `moecp/conformal.py` | `MoECP` (Alg. 2: multinomial randomization, divergence weights, weighted quantile) and plain `SplitCP` (used by the tests only) |
| `moecp/data.py` | 3-domain synthetic generator; UCI Bike-sharing / Temperature loaders (auto-download) |
| `moecp/metrics.py` | coverage, length, kNN local coverage, unbiased worst-slice coverage |
| `experiments/synthetic.py` | Sec. 4.1 (Figs. 2–4) |
| `experiments/real.py` | Sec. 4.2 (Fig. 5) |
| `experiments/plot.py` | figures in `figures/` and `results/summary.csv` |
| `moecp/baselines/` | `qrf.py` (quantile-forest wrapper), `cqr.py`, `chr.py`, `cc.py` (SCP+CC), `rlcp.py`, `pcp.py` |
| `../report/` (next to this folder) | `moecp_reproduction.tex` (write-up: algorithm, data, settings, results vs. paper); `extract_paper_targets.py` reads the paper's boxplot medians from its PDF into `paper_targets.csv`; `make_tables.py` builds `tables.tex` |
| `tests/test_conformal.py` | quantile correctness, KL≡CE weights, τ→0 ≈ SCP, Theorem-1 coverage simulation |
| `tests/test_baselines.py` | coverage simulations for CQR, RLCP, SCP+CC and CHR; SCP+CC with an intercept-only basis ≡ split CP; CHR nesting; PCP smoke test |

## Run
```bash
pip install -r requirements.txt
python tests/test_conformal.py                         # ~5 s
python tests/test_baselines.py                         # ~2 min
python experiments/synthetic.py                        # 50 reps, all methods
python experiments/synthetic.py --setting 2
python experiments/real.py --dataset bike              # 50 reps; PCP dominates the runtime
python experiments/real.py --dataset temperature
python experiments/plot.py
python ../report/make_tables.py                        # paper-vs-ours tables for the report
(cd ../report && pdflatex moecp_reproduction.tex)
```
Flags: `--reps`, `--n_jobs`, `--tau`, `--alpha`, `--n`, `--methods CQR CHR SCP+CC RLCP PCP MoECP` (any subset).
`synthetic.py --setting 2` gives the balanced case.

## Settings (from the paper)
| | Synthetic | Real data |
|---|---|---|
| train / cal / test | 500 / 500 / 500 | 1500 / 1500 / 1500 |
| experts K | 3 | 2 |
| hidden layers (experts and gate) | (32, 32, 32) | (64, 64, 64) |
| epochs | 600 | 2000 |
| τ, divergence | 150, KL | 100, KL |

Shared settings: α = 0.1, 50 reps, ReLU, dropout 0.1, batch size 64, learning rate 1e-4.

## Deviations and assumptions
These are places where the paper is not specific.
- **Optimizer and stopping.** Adam. All epochs are run on the whole training split. The paper says only "early stopping ceiling on epochs"; its runtimes (linear in sample size, the same on both real datasets) indicate the ceiling is always reached. `MoERegressor(val_frac=0.1)` turns on early stopping on a held-out slice (patience 100, best weights restored), which under-trains the model: MoE-CP came out 5–28% longer than the paper with it.
- **Bike features.** Uses `hour.csv` (17,379 rows). The 13 features are every column except `dteday`, `casual`, `registered` and the target `cnt`: instant, season, yr, mnth, hr, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed. The paper says 17,389 rows and does not list the features.
- **Temperature.** Rows with missing values are dropped. Target is `Next_Tmin`, with 21 predictors.
- **Standardization.** X and y are standardized using training-split statistics, so lengths are on the standardized-y scale. Synthetic data is left in raw units.
- **Infinite intervals.** MoE-CP returns (−∞, ∞) when no calibration point has a gating vector close to the test point's, because the test point's own weight then exceeds α. This happens at the extremes of the gate, for example X ≈ ±3 in the synthetic data. `length` is the mean over finite intervals; `inf_frac` and `median_length` are reported alongside it.
- **Worst-slice coverage.** The unbiased version: 20% of the test set is used to search for the slab and 80% to evaluate it, with δ = 0.1 and 1000 random directions. This matches the PCP repo's `wsc_unbiased`.
- **Infinite intervals in the baselines.** RLCP and PCP can also return (−∞, ∞). The PCP repo replaces these by the largest test residual (`finite=True`); here they are excluded from `length` and counted in `inf_frac`, the same as for MoE-CP.
- **Quantile forest.** `quantile-forest` replaces the defunct `skgarden`.
- **SCP+CC.**
  - Set up as `CC` in the PCP repo: signed score y − μ̂(x), linear basis (1, x), lower bound at level α/2 and upper bound at 1 − α/2, both randomized.
  - The reference calls the `conditionalconformal` package, which does not run on NumPy ≥ 2.4. Here each cutoff is one LP (a tilted quantile regression) solved with SciPy. Given the same thresholds, the two agree to 1e-14 in most cutoffs; where they differ (up to 0.06 on a 14-column basis) the LP is the one that meets the method's definition.
  - The two bounds can cross. Such a point counts as not covered, with length |upper − lower|, as in the reference.
- **CHR.**
  - The y-range is the training-split min/max ± 10%. The reference uses min/max of all Y, including test data.
  - No tail smoothing: below the 0.01 and above the 0.99 quantile, the CDF is linear out to the range ends.
  - Among equally short windows, the one with the most mass is chosen.
  - The calibrated level uses the standard ⌈(1−α)(n+1)⌉ order statistic.
  - The construction is vectorized across points; the algorithm is the same.
- **PCP.**
  - The calibration-time refit adds the test point to each fold's classifier with an exact rank-one update. The reference's update mutates its cached inverses in place, so changes accumulate across test points, and its denominator uses xᵀAx where the formula needs xᵀA⁻¹x.
  - The k-means++ restart is selected by true inertia; the reference sums argmin indices.
  - Training residuals are 10-fold cross-validated residuals of the RF.

## Results (50 reps, α = 0.1; medians, paper's value in parentheses)
The paper reports boxplots only; its values are the box medians read from the vector figures in the PDF (`../report/extract_paper_targets.py`). "Length" counts an infinite interval as 2 × the largest test residual of the method's predictor, the PCP-repo convention; the shortest is in bold. "Inf" is the mean share of infinite intervals. Bike and Temperature are on the standardized-y scale. Time is seconds per rep on one core, including base-model training. The write-up is in `../report/moecp_reproduction.pdf`.

| | CQR | CHR | SCP+CC | RLCP | PCP | **MoE-CP** |
|---|---|---|---|---|---|---|
| **Synthetic S1:** coverage | 0.897 (0.897) | 0.894 (0.898) | 0.900 (0.902) | 0.907 (0.909) | 0.923 (0.944) | 0.907 (0.911) |
| length | 13.73 (13.32) | 13.06 (12.82) | 21.20 (22.46) | 16.25 (16.79) | 17.44 (21.12) | **11.84** (11.74) |
| inf % | 0.0 | 0.0 | 0.0 | 0.0 | 2.7 | 0.3 |
| **Synthetic S2:** coverage | 0.899 (0.906) | 0.897 (0.908) | 0.894 (0.897) | 0.908 (0.910) | 0.926 (0.944) | 0.906 (0.911) |
| length | 13.01 (12.67) | 12.17 (11.91) | 19.42 (19.36) | 15.87 (16.00) | 17.00 (20.64) | **11.88** (11.91) |
| inf % | 0.0 | 0.0 | 0.0 | 0.0 | 3.4 | 0.2 |
| **Bike:** coverage | 0.903 (0.906) | 0.898 (0.907) | 0.899 (0.902) | 0.916 (0.920) | 0.911 (0.920) | 0.900 (0.906) |
| length | 1.860 (1.785) | 1.690 (1.610) | 1.072 (1.116) | 1.555 (1.588) | 1.227 (1.302) | **1.054** (1.077) |
| worst-slice coverage | 0.879 (0.883) | 0.891 (0.892) | 0.895 (0.884) | 0.905 (0.901) | 0.906 (0.910) | 0.855 (0.885) |
| inf % | 0.0 | 0.0 | 0.0 | 11.6 | 3.0 | 0.1 |
| **Temperature:** coverage | 0.901 (0.902) | 0.899 (0.905) | 0.901 (0.902) | 0.925 (0.928) | 0.913 (0.920) | 0.902 (0.903) |
| length | 1.489 (1.447) | 1.476 (1.439) | 1.196 (1.189) | 1.949 (1.852) | 1.361 (1.352) | **1.073** (1.067) |
| worst-slice coverage | 0.886 (0.885) | 0.892 (0.892) | 0.889 (0.891) | 0.921 (0.917) | 0.902 (0.915) | 0.885 (0.903) |
| inf % | 0.0 | 0.0 | 0.0 | 27.4 | 5.1 | 0.2 |
| **Time per rep (s):** Bike | 0.4 | 13 | 75 | 1 | 158 | 89 |

How this compares with the paper:
- **MoE-CP** is within 2.1% of the paper's median length on all four experiments, at matching coverage, and is the shortest method in each, as the paper reports. It is the shortest in 90% (S1), 60% (S2), 64% (Bike) and 100% (Temperature) of reps.
- **Ordering.** The methods rank by length exactly as in the paper on S1, Bike and Temperature. On S2 only PCP and SCP+CC swap places.
- **Worst-slice coverage of MoE-CP** is lower than the paper's on both real datasets (0.855 vs 0.885, 0.885 vs 0.903), though inside the paper's interquartile range.
- **CQR and CHR** are 2–5% longer than the paper's. The quantile-forest settings and the Bike feature set were ruled out as the cause; it is unexplained.
- **PCP on synthetic data** is less conservative than the paper's (coverage 0.92 vs 0.94, length 17% shorter). A possible cause is the three places listed above where this implementation corrects the reference code; that has not been verified.
- RLCP and PCP over-cover (0.91–0.93) and produce many infinite intervals, matching the paper's remark that they are "slightly conservative".
