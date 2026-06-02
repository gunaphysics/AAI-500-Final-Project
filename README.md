# Predicting 30-Day Heart Failure Readmission Risk

**AAI-500 — Probability and Statistics for Artificial Intelligence**
Shiley-Marcos School of Engineering, University of San Diego
Group 4 — Final Team Project

## Team

| Member | Role |
|---|---|
| Ian Schmitt | Team lead |
| Keana Gindlesperger | Member |
| Guna Pasupathy | Member |

Upstream repository: <https://github.com/KeanaGindy/AAI-500-FinalProject>

## Project Overview

We analyze factors associated with 30-day heart failure readmission and build a
defensible risk-prediction workflow using clinical, treatment, and social
determinant data.

**Objectives**

1. Clean and validate the dataset.
2. Perform exploratory data analysis and statistical testing.
3. Compare interpretable models and machine-learning classifiers against a
   majority-class baseline.
4. Translate results into practical recommendations and clearly document
   limitations.

## Dataset

**Source.** Kaggle — [*Heart Failure Readmission and SDOH Dataset*](https://www.kaggle.com/datasets/nudratabbas/heart-failure-readmission-and-sdoh-dataset)
(`nudratabbas/heart-failure-readmission-and-sdoh-dataset`). 3,000 patient
records × 16 variables. The raw CSV is **not** committed. You do **not** need to
download it by hand — `src.data.load.load_raw()` pulls it from Kaggle via
`kagglehub` on first use and caches a copy at
`data/raw/heart_failure_readmission_dataset.csv`. This requires Kaggle API
credentials (see [Setup](#setup)). The slug and filename live in `src/config.py`
(`KAGGLE_DATASET`, `RAW_FILENAME`).

**Variable groups**

| Group | Variables |
|---|---|
| Identifier (excluded from modeling) | `patient_id` |
| Clinical | age, gender, BMI, BNP, sodium, creatinine, systolic BP, heart rate |
| Treatment | ACE inhibitor, beta blocker, diuretic, adherence score |
| Social determinants | income level, distance to hospital (km) |
| Target | `readmitted_30d` (~41% positive class) |

**Known data-quality issues** (from the initial EDA reported in the proposal)

- ~3% missing values in BMI, sodium, and creatinine.
- Impossible negative creatinine values.
- Several extreme clinical outliers that require review before modeling.

`src/data/clean.py` codifies clinically plausible ranges for each variable and
converts out-of-range readings to `NaN` so downstream imputation can handle them.

## Repository Structure

```
.
├── data/
│   ├── raw/                # untracked — Kaggle CSV auto-downloaded here
│   ├── interim/            # intermediate artifacts (untracked)
│   └── processed/          # modeling-ready files (untracked)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_cleaning.ipynb
│   ├── 03_statistical_testing.ipynb
│   └── 04_modeling.ipynb
├── src/
│   ├── config.py           # paths, constants, random seed
│   ├── data/               # load and clean
│   ├── features/           # preprocessing pipelines
│   ├── models/             # baseline, candidate models, evaluation
│   └── viz/                # shared plotting helpers
├── tests/                  # pytest unit tests for src/
├── reports/figures/        # generated charts (untracked)
├── models/                 # serialized model artifacts (untracked)
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Clone and enter the project
git clone https://github.com/KeanaGindy/AAI-500-FinalProject.git
cd AAI-500-FinalProject

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure Kaggle API credentials (one-time)
#    Kaggle → Account → "Create New API Token" downloads kaggle.json.
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
#    (Alternatively export KAGGLE_USERNAME and KAGGLE_KEY.)
```

The dataset is pulled automatically the first time `load_raw()` runs — no manual
download needed. To fetch it ahead of time:

```bash
python -c "from src.data.load import load_raw; print(load_raw().shape)"
```

The notebooks add the project root to `sys.path`, so `from src.config import ...`
works without an editable install.

## Running the Analysis

```bash
jupyter notebook
```

Run the notebooks in order:

| # | Notebook | Purpose |
|---|---|---|
| 01 | `01_eda.ipynb` | Load the raw data, summarize distributions, visualize missingness and target balance. |
| 02 | `02_cleaning.ipynb` | Drop the identifier, flag clinically impossible values, save the cleaned frame to `data/processed/`. |
| 03 | `03_statistical_testing.ipynb` | Chi-square tests for categorical predictors and Welch's t-tests for numeric predictors against the readmission target, with a Bonferroni correction for multiple testing. |
| 04 | `04_modeling.ipynb` | Fit a logistic-regression GLM (`statsmodels`); report odds ratios with confidence intervals and compare models with likelihood-ratio tests, deviance, and AIC. An optional appendix runs tree/ensemble models for comparison. |
| 05 | `05_eda_detailed.ipynb` | In-depth EDA covering Ch 1–5 (descriptive stats, distribution fits, CLT, confidence intervals, bootstrap, Bayesian proportion, correlation, and significance tests with effect sizes). Saves all figures and tables to `reports/`. |

The detailed EDA in notebook 05 writes its outputs to `reports/`: charts to
`reports/figures/`, tables to `reports/tables/`, and a written summary at
[`reports/eda_report.md`](reports/eda_report.md).

## Testing

```bash
pytest
```

## Methodology Notes

- **Statistical testing.** Chi-square tests of independence for categorical
  predictors and Welch's t-tests for numeric predictors, with a Bonferroni
  correction controlling the family-wise error rate across all tests.
- **Model.** A logistic-regression GLM fit with `statsmodels`. Numeric
  predictors are standardized so the odds ratios are comparable; categorical
  predictors enter as indicator variables. Missing labs are handled by
  complete-case analysis.
- **Model comparison.** Likelihood-ratio tests and AIC (full vs. null vs. a
  reduced model), with a majority-class baseline for the accuracy reference.
- **Out of scope.** Tree/ensemble models and ROC-AUC live in a clearly
  labeled appendix in `04_modeling`; they are not used for any conclusions.
- **Reproducibility.** A single `RANDOM_STATE` in `src/config.py` is reused
  wherever a split is taken.

## Limitations (working list)

- Single-source observational data; selection biases in who is admitted and
  re-admitted are not directly addressed.
- The dataset has no time-to-event information, so survival analysis is out
  of scope.
- Social determinants are coarsely encoded (income bucket, distance to
  hospital) and likely under-represent the underlying construct.
- This is a coursework analysis. Results are not deployment-ready and any
  clinical interpretation should be validated independently.

## References

- University of San Diego, AAI-500 *Probability and Statistics for AI*.
- Kaggle — *Heart Failure Readmission and Social Determinants of Health* dataset.
- scikit-learn, pandas, statsmodels, scipy documentation.
