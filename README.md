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

**Source.** Kaggle — *Heart Failure Readmission and Social Determinants of
Health Dataset*. 3,000 patient records × 16 variables. The raw CSV is **not**
committed; download it from Kaggle and drop it into `data/raw/`. The default
filename expected by `src/config.py` is `heart_failure_readmission.csv` —
update `RAW_FILENAME` there if your file is named differently.

**Variable groups**

| Group | Variables |
|---|---|
| Identifier (excluded from modeling) | `patient_id` |
| Clinical | age, BMI, BNP, sodium, creatinine, blood pressure, heart rate |
| Treatment | ACE inhibitor, beta blocker, diuretic |
| Social determinants | income level, distance to hospital |
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
│   ├── raw/                # untracked — place the Kaggle CSV here
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

# 4. Place the raw Kaggle CSV at data/raw/heart_failure_readmission.csv
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
| 03 | `03_statistical_testing.ipynb` | Chi-square tests for categorical predictors; t-tests / Mann–Whitney for numeric predictors against the readmission target. |
| 04 | `04_modeling.ipynb` | Train baseline, logistic regression, decision tree, random forest, and gradient boosting; compare on accuracy, precision, recall, F1, and ROC-AUC. |

## Testing

```bash
pytest
```

## Methodology Notes

- **Preprocessing.** Median-impute + standard-scale numeric features;
  mode-impute + one-hot-encode categorical features. Wired through a single
  `ColumnTransformer` so the test split never leaks into training statistics.
- **Validation.** Stratified hold-out split with the seed defined in
  `src/config.py`; extend with cross-validation as needed in `04_modeling`.
- **Baseline.** A majority-class `DummyClassifier`. Models that fail to beat
  it on ROC-AUC should not be reported as predictive.
- **Reproducibility.** A single `RANDOM_STATE` is defined in `src/config.py`
  and propagated to every estimator and split.

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
