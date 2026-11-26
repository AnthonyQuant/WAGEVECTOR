# WAGEVECTOR — Reproducible Econometric & Statistical Wage Modeling

A leakage-free, end-to-end statistical modeling and econometric inference pipeline analyzing worker compensation determinants in the classic ISLR `Wage` dataset (`3,000` male workers in the Middle Atlantic region, `2003–2009`).

---

## 1. Research Questions

- **RQ1 (Multivariable Predictive Gain & Out-of-Time Stability):** How much out-of-sample predictive accuracy (`R²`, `MAE`, `RMSE`) is gained by moving from a zero-feature training-mean benchmark (**Model 0**) and a single-variable age regression (**Model 1**) to a full-rank `13`-predictor multivariable specification (**Model 2** & **Model 3**), and do those relationships hold on an unseen `2009` temporal holdout?
- **RQ2 (Conditional Wage Determinants & Robust Inference):** Holding survey year, age, education, job sector, marital status, self-reported health, and health insurance coverage simultaneous, which predictors carry statistically significant conditional wage premia or penalties under both classical OLS and MacKinnon–White **HC3** heteroskedasticity-consistent standard errors?
- **RQ3 (Log-Target Stabilization & Retransformation Bias):** How effectively does modeling `logwage = ln(wage)` reduce residual skewness, tail kurtosis, and heteroskedasticity, and how much Jensen's-inequality downward mean bias occurs under naive exponentiation `exp(z_hat)` versus Duan's non-parametric training-set smearing `S_Duan * exp(z_hat)`?

---

## 2. Repository Structure

```text
WAGEVECTOR/
├── data/
│   ├── raw/
│   │   └── Wage.csv                  # Immutable source dataset (3,000 rows x 13 cols)
│   └── processed/
│       └── wage_cleaned.csv          # Full-rank encoded modeling table (3,000 rows x 15 cols)
├── figures/
│   ├── wage_distribution.png         # Figure 1 (300 DPI): Raw wage vs. logwage distributions
│   ├── wage_relationships.png        # Figure 2 (300 DPI): Age & year conditional wage trends
│   ├── categorical_analysis.png      # Figure 3 (300 DPI): Wage & logwage across 5 categorical factors
│   ├── residual_diagnostics.png      # Figure 4 (300 DPI): Residuals vs. fitted, Scale-Location & Normal Q-Q
│   ├── model_comparison.png          # Figure 5 (300 DPI): Held-out R², RMSE/MAE & retransformation bias
│   └── coefficient_effects.png       # Figure 6 (300 DPI): Multivariable slopes & exact % shifts (95% CI)
├── results/
│   ├── model_comparison.csv          # Train/test metrics across Models 0, 1, 2, 3 (Native, Naive, Duan)
│   └── model_coefficients.csv        # Classical OLS & HC3-robust SEs, t-stats, p-values & 95% CIs
├── notebook/
│   └── WAGEVECTOR.ipynb              # Canonical end-to-end executable notebook (Steps 1–33)
├── README.md                         # Project documentation, methodology & empirical summary
└── LICENSE                           # Optional repository license
```

---

## 3. Methodology & Anti-Leakage Design

1. **Feature Pruning & Full-Rank Reference Encoding:** Drops `Unnamed: 0` (row index) and zero-variance/uninformative columns (`sex`, `region`, `race` where `sex == '1. Male'` and `region == '2. Middle Atlantic'` for 100% of records). Encodes the `5` categorical predictors using `drop_first=True` anchored to explicit baseline categories (`1. < HS Grad`, `1. Industrial`, `1. Never Married`, `1. <=Good`, `1. Yes`), yielding `13` full-rank predictors (`rank = 14 / 14` with intercept).
2. **Shared Deterministic Split (`RANDOM_STATE = 42`):** Partitions `wage_cleaned.csv` (`3,000 × 15`) into `n_train = 2,400` (`80.0%`) and `n_test = 600` (`20.0%`) with strict isolation of `wage` and `logwage` from `X` and verified positive representation across all `11` dummy indicators in both folds.
3. **Model Suite (`Models 0–3`):**
   - **Model 0:** Zero-feature training-mean benchmark (`wage_hat = mean(y_wage_train)`).
   - **Model 1:** Simple Age-Only OLS (`wage ~ age`).
   - **Model 2:** Multivariable Raw-Wage OLS (`wage ~ X_all`, `13` predictors).
   - **Model 3:** Multivariable Log-Target OLS (`logwage ~ X_all`, `13` predictors), evaluated both on its native `ln(wage)` scale and retransformed to raw `wage` units via naive `exp(z_hat)` (**Model 3A**) and train-only Duan smearing `1.040896 * exp(z_hat)` (**Model 3B**).
4. **Scale Separation & Heteroskedasticity Guardrails:** Natural-log `R²` and `RMSE` (**Model 3 Native**) are segregated from raw-wage metrics so cross-scale numbers are never conflated. Because Breusch–Pagan LM tests reject strict homoskedasticity (`p < 0.001`), `results/model_coefficients.csv` reports both classical OLS standard errors and MacKinnon–White **HC3** robust standard errors.

---

## 4. Key Empirical Findings

### A. Out-of-Sample Model Benchmark (`n_train = 2,400`, `n_test = 600`)

| Model ID | Specification | Evaluation Scale | Train R² | Test R² | Test MAE | Test RMSE | Test Mean Resid |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Model 0 | `wage_hat = mean(y_wage_train)` | Raw wage | +0.0000 | -0.0006 | 29.1025 | 39.4959 | -0.9533 |
| Model 1 | `wage ~ age` | Raw wage | +0.0407 | +0.0262 | 28.7688 | 38.9630 | -1.3757 |
| Model 2 | `wage ~ X_all (13 predictors)` | Raw wage | +0.3279 | +0.3812 | 22.0875 | 31.0608 | +0.1312 |
| Model 3 (Native) | `logwage ~ X_all (13 predictors)` | ln(wage) | +0.3636 | +0.4172 | 0.1935 | 0.2531 | +0.0077 |
| Model 3A (Naive Exp) | `exp(logwage_hat) [uncorrected Jensen bias]` | Raw wage | +0.3179 | +0.3743 | 21.5888 | 31.2333 | +4.4948 |
| Model 3B (Duan Smeared) | `1.040696 * exp(logwage_hat) [train smearing]` | Raw wage | +0.3301 | +0.3891 | 21.8396 | 30.8598 | +0.1629 |

- **Multivariable Gain (`RQ1`):** Adding `age` alone lifts held-out `R²` from `-0.0006` (**Model 0**) to `+0.0262` (**Model 1**), whereas adding the full `13`-predictor socioeconomic and health-insurance feature set achieves `Test R² = +0.3812` (**Model 2**, `RMSE = 31.0608`) and `Test R² = +0.3891` (**Model 3B Duan Smeared**, `RMSE = 30.8598`, `MAE = 21.8396`, cutting `RMSE` by `21.87%` vs. baseline). Under an isolated `2003–2008` train / `2009` test temporal holdout (`n_2009 = 389`), multivariable models retain `Test R² = 0.3067` (**Model 2**) and `0.3155` (**Model 3B**).
- **Log-Target Stabilization & Duan Smearing (`RQ3`):** Modeling `logwage = ln(wage)` (`Train R² = 0.3636`, `Test R² = 0.4172`, `Test RMSE = 0.2531` in `ln(wage)` units) reduces training residual skewness from `+1.7209` to `-0.3509` and cuts the Jarque–Bera normality statistic by `88.5%` (`5,511.84 → 586.50`). Retransforming via Duan's training-only smearing factor (`S_Duan = 1.040896`) eliminates `96.4%` of naive exponentiation's downward mean bias (`+4.4948 → +0.1629` recorded wage units).

### B. Conditional Predictor Effects & HC3 Inference (`Train n = 2,400`, `Max VIF = 3.4193`)

| Predictor Term | Reference / Unit | Model 2 Slope (Raw) | Model 2 `p` (HC3) | Model 3 Slope (`ln`) | Model 3 Exact `%` Shift | Model 3 Exact `95% CI` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `year` | Per +1 nominal survey year (unadjusted) | +1.1644 | 0.001291 | +0.012171 | +1.22% | `[+0.65%, +1.80%]` |
| `age` | Per +1 year of worker age | +0.3295 | 2.717e-06 | +0.003107 | +0.31% | `[+0.20%, +0.43%]` |
| `education_2. HS Grad` | Baseline: '1. < HS Grad' | +7.2271 | 2.929e-05 | +0.076732 | +7.98% | `[+3.23%, +12.94%]` |
| `education_3. Some College` | Baseline: '1. < HS Grad' | +17.0413 | 1.94e-16 | +0.170414 | +18.58% | `[+13.03%, +24.40%]` |
| `education_4. College Grad` | Baseline: '1. < HS Grad' | +31.6432 | 6.175e-43 | +0.283092 | +32.72% | `[+26.50%, +39.25%]` |
| `education_5. Advanced Degree` | Baseline: '1. < HS Grad' | +52.6049 | 3.561e-55 | +0.420852 | +52.33% | `[+44.53%, +60.54%]` |
| `jobclass_2. Information` | Baseline: '1. Industrial' | +3.1408 | 0.02967 | +0.021115 | +2.13% | `[-0.30%, +4.63%]` |
| `maritl_2. Married` | Baseline: '1. Never Married' | +16.7738 | 1.359e-21 | +0.161802 | +17.56% | `[+13.94%, +21.30%]` |
| `maritl_3. Widowed` | Baseline: '1. Never Married' | +0.6846 | 0.9135 | +0.043284 | +4.42% | `[-9.56%, +20.57%]` |
| `maritl_4. Divorced` | Baseline: '1. Never Married' | +3.2764 | 0.2528 | +0.037372 | +3.81% | `[-1.45%, +9.35%]` |
| `maritl_5. Separated` | Baseline: '1. Never Married' | +10.9417 | 0.01704 | +0.117313 | +12.45% | `[+2.46%, +23.41%]` |
| `health_2. >=Very Good` | Baseline: '1. <=Good' | +7.1378 | 2.159e-06 | +0.064507 | +6.66% | `[+3.90%, +9.50%]` |
| `health_ins_2. No` | Baseline: '1. Yes' (Insured) | -17.1244 | 1.888e-26 | -0.192713 | -17.53% | `[-19.64%, -15.36%]` |

- **Dominant Socioeconomic Drivers (`RQ2`):** `10/13` predictors are statistically significant at `α = 0.05` under both classical and HC3 inference. Returns to education increase strictly monotonically over `1. < HS Grad`, reaching `+52.60` raw wage units (`+52.33%`, `p < 0.001`) for `5. Advanced Degree`, while lacking health insurance (`health_ins_2. No`) is associated with `-17.12` units (`-17.53%`, `p < 0.001`) relative to insured workers (`1. Yes`).

---

## 5. Publication Figures (`300 DPI`)

1. `figures/wage_distribution.png` — Distribution of raw `wage` (skewness `+1.67`, median `104.92`, mean `111.70`) vs. variance-stabilized `logwage = ln(wage)`.
2. `figures/wage_relationships.png` — Continuous relationships of `wage` and `logwage` across worker `age` (concave lifecycle profile) and nominal survey `year` (`2003–2009`).
3. `figures/categorical_analysis.png` — Unconditional subgroup distributions across `education`, `jobclass`, `maritl`, `health`, and `health_ins`.
4. `figures/residual_diagnostics.png` — Side-by-side Residuals vs. Fitted, Scale-Location spread, and Standardized Normal Q-Q comparison between **Model 2** and **Model 3**.
5. `figures/model_comparison.png` — Held-out `R²`, `RMSE`/`MAE`, and signed mean residual bias across **Models 0, 1, 2, 3A (Naive Exp), and 3B (Duan Smeared)**.
6. `figures/coefficient_effects.png` — Forest plot of conditional raw-wage slopes (**Model 2**) and exact multiplicative percentage shifts (**Model 3**) with `95%` confidence intervals.

---

## 6. Reproducibility & Execution

1. Ensure `WAGEVECTOR/data/raw/Wage.csv` (`3,000 × 13`) is present in the project root.
2. Open `WAGEVECTOR/notebook/WAGEVECTOR.ipynb` in Google Colab or Jupyter and run all cells top-to-bottom (`Runtime -> Run all`).
3. All random partitioning uses `RANDOM_STATE = 42`, regenerating `data/processed/wage_cleaned.csv`, `results/model_comparison.csv`, `results/model_coefficients.csv`, and all six `300 DPI` figures to bit-exact equivalence (`max |Δ| <= 1e-12`).

---

## 7. Methodological & Scope Limitations

- **Associational / Non-Causal Scope:** Observational CPS cross-sectional records omit unobserved ability, detailed occupation codes, union membership, tenure, and firm size; all slopes represent conditional associations rather than causal parameters.
- **Demographic & Geographic Sample Boundary:** Covers `3,000` male workers in the Middle Atlantic region (`2003–2009`), preventing direct generalization to female workers, other geographic regions, or post-2009 labor markets.
- **Nominal Wage Units:** Recorded `wage` values are unadjusted for CPI inflation, so the `year` coefficient reflects nominal wage drift alongside the `2008–2009` macroeconomic plateau.
- **Conditional Heteroskedasticity:** Breusch–Pagan LM tests reject strict constant variance (`p < 0.001`), motivating our joint reporting of classical OLS and MacKinnon–White **HC3** robust standard errors.

---

## 8. Data Attribution

Source dataset: `Wage` from *An Introduction to Statistical Learning* (`ISLR`, James, Witten, Hastie, and Tibshirani), compiled from the Current Population Survey (CPS) for the Middle Atlantic region (`2003–2009`).
