import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from pathlib import Path
from pipeline_utils import (
    find_data_root,
    safe_to_csv,
    validate_patient_ids,
)


# ============================================================
# PATH CONFIGURATION
# ------------------------------------------------------------
# This script creates one fixed pseudo-outcome file from the
# fixed semi-synthetic training data.
# ============================================================

DATA_ROOT = find_data_root()
SEMISYN_DIR = DATA_ROOT / "01_semisynthetic_data"
PSEUDO_DIR = DATA_ROOT / "02_dr_pseudo_outcomes"

TRAIN_DATA_PATH = SEMISYN_DIR / "IST_syn_train.csv"
OUTPUT_PATH = PSEUDO_DIR / "pseudo_outcomes_train.csv"

# Fixed artifacts are protected by default.
OVERWRITE_EXISTING = False


# ============================================================
# 1. Load semi-synthetic train data
# ============================================================

train_df = pd.read_csv(TRAIN_DATA_PATH)
validate_patient_ids(train_df, "Training data")

print("Train dataset shape:", train_df.shape)
print("\nFirst five rows:")
print(train_df.head())


# ============================================================
# 2. Select observed variables for the DR learner
# ============================================================

feature_columns = [
    "AGE",
    "SEX",
    "RSBP",
    "RATRIAL",
    "RVISINF",
    "RHEP24",
    "RASP3",
    "RDEF1",
    "RDEF2",
    "RDEF3",
    "RDEF4",
    "RDEF5",
    "RDEF6",
    "RDEF7",
    "RDEF8",
]

required_columns = ["patient_id", "A", "Y"] + feature_columns
missing_columns = [
    column
    for column in required_columns
    if column not in train_df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

X_raw = train_df[feature_columns].copy()
A = train_df["A"].copy()
Y = train_df["Y"].copy()

if not set(A.unique()).issubset({0, 1}):
    raise ValueError("A must contain only 0 and 1.")

print("\nRaw X shape:", X_raw.shape)
print("A shape:", A.shape)
print("Y shape:", Y.shape)


# ============================================================
# 3. Convert X into numeric features
# ============================================================

numeric_columns = ["AGE", "RSBP"]

categorical_columns = [
    column
    for column in feature_columns
    if column not in numeric_columns
]

X = pd.get_dummies(
    X_raw,
    columns=categorical_columns,
    dtype=int,
)

continuous_features = ["AGE", "RSBP"]
X_means = X[continuous_features].mean()
X_stds = X[continuous_features].std()

if (X_stds == 0).any():
    raise ValueError("At least one continuous feature has zero variance.")

X[continuous_features] = (
    X[continuous_features] - X_means
) / X_stds

print("\nMeans after normalization:")
print(X[continuous_features].mean())
print("\nStandard deviations after normalization:")
print(X[continuous_features].std())


# ============================================================
# 4. Fit models for the DR pseudo-outcome
# ============================================================

propensity_model = LogisticRegression(max_iter=1000)
propensity_model.fit(X, A)
e_hat = propensity_model.predict_proba(X)[:, 1]

treated_mask = A == 1
control_mask = A == 0

if treated_mask.sum() == 0 or control_mask.sum() == 0:
    raise ValueError("Both treatment groups are needed for DR pseudo-outcomes.")

mu0_model = LinearRegression()
mu0_model.fit(X.loc[control_mask], Y.loc[control_mask])
mu0_hat = mu0_model.predict(X)

mu1_model = LinearRegression()
mu1_model.fit(X.loc[treated_mask], Y.loc[treated_mask])
mu1_hat = mu1_model.predict(X)

print("\nEstimated propensity summary:")
print(pd.Series(e_hat).describe())


# ============================================================
# 5. Construct the DR pseudo-outcome
# ============================================================

pseudo_outcome = (
    mu1_hat
    - mu0_hat
    + A.to_numpy() * (Y.to_numpy() - mu1_hat) / e_hat
    - (1 - A.to_numpy()) * (Y.to_numpy() - mu0_hat) / (1 - e_hat)
)

if not np.isfinite(pseudo_outcome).all():
    raise ValueError("Pseudo-outcome contains NaN or infinite values.")

pseudo_outcomes_df = pd.DataFrame(
    {
        "patient_id": train_df["patient_id"],
        "pseudo_outcome": pseudo_outcome,
        "e_hat": e_hat,
        "mu0_hat": mu0_hat,
        "mu1_hat": mu1_hat,
    }
)

validate_patient_ids(pseudo_outcomes_df, "Pseudo-outcomes")

if len(pseudo_outcomes_df) != len(train_df):
    raise ValueError("Pseudo-outcome row count does not match training data.")

print("\nPseudo-outcome summary:")
print(pseudo_outcomes_df["pseudo_outcome"].describe())


# ============================================================
# 6. Save fixed pseudo-outcomes
# ============================================================

safe_to_csv(
    pseudo_outcomes_df,
    OUTPUT_PATH,
    OVERWRITE_EXISTING,
)

print("\nTarget pseudo-outcome file:")
print(OUTPUT_PATH)
