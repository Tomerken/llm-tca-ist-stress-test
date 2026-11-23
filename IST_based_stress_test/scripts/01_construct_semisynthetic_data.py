import os
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ------------------------------------------------------------
# This script builds the fixed semi-synthetic dataset used by the submitted experiment.
# ============================================================

def _find_data_root(marker: str = "IST_based_stress_test") -> Path:
    env = os.environ.get("IST_DATA_ROOT")

    if env:
        data_root = Path(env).expanduser().resolve()

        if data_root.is_dir():
            return data_root

        raise FileNotFoundError(
            f"IST_DATA_ROOT={env!r} is not a directory."
        )

    start = Path(__file__).resolve()

    for parent in (start.parent, *start.parents):
        if parent.name == marker and parent.is_dir():
            return parent

        candidate = parent / marker

        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        f"Could not locate a '{marker}' directory above {start}. "
        f"Set IST_DATA_ROOT to point at it."
    )


DATA_ROOT = _find_data_root()

RAW_DIR = DATA_ROOT / "00_raw"
SEMISYN_DIR = DATA_ROOT / "01_semisynthetic_data"

DATA_PATH = RAW_DIR / "IST_corrected.csv"
VARIABLES_PATH = RAW_DIR / "IST_variables.csv"

TRAIN_OUTPUT_PATH = SEMISYN_DIR / "IST_syn_train.csv"
TEST_OUTPUT_PATH = SEMISYN_DIR / "IST_syn_test.csv"
DGP_SPEC_OUTPUT_PATH = SEMISYN_DIR / "dgp_spec.csv"

# Existing fixed data artifacts are protected by default.
OVERWRITE_EXISTING = False


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_to_csv(
    data_frame: pd.DataFrame,
    output_path: Path,
    overwrite: bool,
) -> bool:
    ensure_dir(output_path.parent)

    if output_path.exists() and not overwrite:
        print(
            "[skip ] exists, not overwriting "
            f"(set OVERWRITE_EXISTING=True): {output_path}"
        )
        return False

    data_frame.to_csv(output_path, index=False)
    print(f"[write] {output_path}")
    return True


# ============================================================
# 1. Load IST and select the text-visible covariates
# ============================================================

raw_df = pd.read_csv(DATA_PATH)

required_x_columns = [
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

missing_columns = [
    column
    for column in required_x_columns
    if column not in raw_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required IST columns: {missing_columns}"
    )

complete_cases = raw_df.dropna(
    subset=required_x_columns,
).copy()

print("Raw IST shape:", raw_df.shape)
print("Complete-case shape:", complete_cases.shape)


# ============================================================
# 2. Create a fixed train/test split
# ------------------------------------------------------------
# This fixed split is used by every downstream step.
# ============================================================

sample_size = 2500
train_size = 2000
random_seed = 42

if len(complete_cases) < sample_size:
    raise ValueError(
        f"Need at least {sample_size} complete rows, "
        f"but found only {len(complete_cases)}."
    )

sampled_df = complete_cases.sample(
    n=sample_size,
    random_state=random_seed,
).reset_index(drop=True)

train_df = sampled_df.iloc[:train_size].copy()
test_df = sampled_df.iloc[train_size:].copy()

train_df.insert(
    0,
    "patient_id",
    range(train_size),
)

test_df.insert(
    0,
    "patient_id",
    range(train_size, sample_size),
)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)


# ============================================================
# 3. Build a numeric covariate matrix
# ------------------------------------------------------------
# Raw clinical columns are preserved in train_df/test_df for text
# generation. X_model is only for the DGP formulas.
# ============================================================

X_raw = sampled_df[required_x_columns].copy()

continuous_features = [
    "AGE",
    "RSBP",
]

categorical_features = [
    column
    for column in required_x_columns
    if column not in continuous_features
]

X_model = pd.get_dummies(
    X_raw,
    columns=categorical_features,
    dtype=float,
)

train_indices = train_df.index
test_indices = test_df.index

X_train_model = X_model.loc[train_indices].copy()
X_test_model = X_model.loc[test_indices].copy()

train_means = X_train_model[continuous_features].mean()
train_stds = X_train_model[continuous_features].std()

if (train_stds == 0).any():
    raise ValueError(
        "At least one continuous feature has zero training variance."
    )

X_train_model[continuous_features] = (
    X_train_model[continuous_features] - train_means
) / train_stds

X_test_model[continuous_features] = (
    X_test_model[continuous_features] - train_means
) / train_stds

print("\nEncoded feature count:", X_model.shape[1])
print("Train-normalized AGE/RSBP means:")
print(X_train_model[continuous_features].mean())
print("Train-normalized AGE/RSBP standard deviations:")
print(X_train_model[continuous_features].std())


# ============================================================
# 4. Fixed DGP: all 15 text-visible covariate groups are active
# ------------------------------------------------------------
# Every raw covariate group affects:
#   1. treatment assignment e(X),
#   2. the baseline outcome Y(0), and
#   3. the heterogeneous treatment effect tau(X).
#
# For RDEF1-RDEF8:
#   N is the reference category;
#   Y and C ("cannot assess") receive separate coefficients.
# ============================================================

PROPENSITY_INTERCEPT = -0.15
BASELINE_INTERCEPT = 1.20
TAU_INTERCEPT = 0.35

# e(X): treatment assignment
PROPENSITY_COEFFICIENTS = {
    "AGE": 0.35,
    "RSBP": 0.25,
    "SEX_M": 0.10,
    "RATRIAL_Y": 0.35,
    "RVISINF_Y": 0.30,
    "RHEP24_Y": 0.22,
    "RASP3_Y": 0.18,
    "RDEF1_Y": 0.16,
    "RDEF1_C": 0.08,
    "RDEF2_Y": 0.22,
    "RDEF2_C": 0.10,
    "RDEF3_Y": 0.18,
    "RDEF3_C": 0.08,
    "RDEF4_Y": 0.26,
    "RDEF4_C": 0.12,
    "RDEF5_Y": 0.16,
    "RDEF5_C": 0.08,
    "RDEF6_Y": 0.14,
    "RDEF6_C": 0.07,
    "RDEF7_Y": 0.18,
    "RDEF7_C": 0.09,
    "RDEF8_Y": 0.12,
    "RDEF8_C": 0.06,
}

# E[Y(0) | X]: baseline outcome under control
BASELINE_COEFFICIENTS = {
    "AGE": 0.45,
    "RSBP": 0.30,
    "SEX_M": 0.12,
    "RATRIAL_Y": 0.48,
    "RVISINF_Y": 0.42,
    "RHEP24_Y": 0.28,
    "RASP3_Y": 0.22,
    "RDEF1_Y": 0.24,
    "RDEF1_C": 0.10,
    "RDEF2_Y": 0.32,
    "RDEF2_C": 0.13,
    "RDEF3_Y": 0.26,
    "RDEF3_C": 0.11,
    "RDEF4_Y": 0.36,
    "RDEF4_C": 0.15,
    "RDEF5_Y": 0.24,
    "RDEF5_C": 0.10,
    "RDEF6_Y": 0.20,
    "RDEF6_C": 0.08,
    "RDEF7_Y": 0.28,
    "RDEF7_C": 0.12,
    "RDEF8_Y": 0.18,
    "RDEF8_C": 0.08,
}

# tau(X): heterogeneous treatment effect / CATE.
# Medical variables omitted in Sparse receive relatively
# substantial coefficients so that their removal is meaningful.
TAU_COEFFICIENTS = {
    "AGE": 0.40,
    "RSBP": 0.15,
    "SEX_M": 0.10,
    "RATRIAL_Y": 0.45,
    "RVISINF_Y": -0.40,
    "RHEP24_Y": 0.30,
    "RASP3_Y": -0.25,
    "RDEF1_Y": 0.20,
    "RDEF1_C": 0.08,
    "RDEF2_Y": 0.25,
    "RDEF2_C": 0.10,
    "RDEF3_Y": -0.20,
    "RDEF3_C": -0.08,
    "RDEF4_Y": 0.40,
    "RDEF4_C": 0.16,
    "RDEF5_Y": -0.18,
    "RDEF5_C": -0.07,
    "RDEF6_Y": 0.15,
    "RDEF6_C": 0.06,
    "RDEF7_Y": -0.15,
    "RDEF7_C": -0.06,
    "RDEF8_Y": 0.10,
    "RDEF8_C": 0.04,
}


def validate_model_columns(
    X: pd.DataFrame,
    coefficient_dict: dict[str, float],
    mechanism_name: str,
) -> None:
    missing = sorted(
        set(coefficient_dict) - set(X.columns)
    )

    if missing:
        raise ValueError(
            f"{mechanism_name} requires missing encoded columns: "
            f"{missing}"
        )


for mechanism_name, coefficient_dict in {
    "propensity mechanism": PROPENSITY_COEFFICIENTS,
    "baseline-outcome mechanism": BASELINE_COEFFICIENTS,
    "CATE mechanism": TAU_COEFFICIENTS,
}.items():
    validate_model_columns(
        X_model,
        coefficient_dict,
        mechanism_name,
    )


def linear_score(
    X: pd.DataFrame,
    intercept: float,
    coefficients: dict[str, float],
) -> pd.Series:
    score = pd.Series(
        intercept,
        index=X.index,
        dtype=float,
    )

    for feature_name, coefficient in coefficients.items():
        score = score + coefficient * X[feature_name]

    return score


def sigmoid(score: pd.Series) -> pd.Series:
    clipped_score = np.clip(score, -30, 30)
    return 1.0 / (1.0 + np.exp(-clipped_score))


# Raw-variable groups are listed separately from encoded columns.
# This avoids the earlier error of trying to index X_model with
# raw categorical names such as SEX or RATRIAL.
active_raw_features = required_x_columns.copy()

active_model_features = sorted(
    set(PROPENSITY_COEFFICIENTS)
    | set(BASELINE_COEFFICIENTS)
    | set(TAU_COEFFICIENTS)
)

print("\nActive raw covariate groups (15):")
print(active_raw_features)

print("\nActive encoded model columns:")
print(active_model_features)

print("\nNumber of active encoded columns:", len(active_model_features))


# ============================================================
# 5. Generate treatment assignment A
# ============================================================

train_propensity_score = linear_score(
    X_train_model,
    PROPENSITY_INTERCEPT,
    PROPENSITY_COEFFICIENTS,
)

test_propensity_score = linear_score(
    X_test_model,
    PROPENSITY_INTERCEPT,
    PROPENSITY_COEFFICIENTS,
)

train_propensity = sigmoid(train_propensity_score)
test_propensity = sigmoid(test_propensity_score)

print("\nTrain propensity summary:")
print(train_propensity.describe())

print("\nTest propensity summary:")
print(test_propensity.describe())

rng = np.random.default_rng(seed=random_seed)

train_df["A"] = rng.binomial(
    n=1,
    p=train_propensity.to_numpy(),
)

test_df["A"] = rng.binomial(
    n=1,
    p=test_propensity.to_numpy(),
)

print("\nTreatment proportion in train:")
print(train_df["A"].value_counts(normalize=True).sort_index())

print("\nTreatment proportion in test:")
print(test_df["A"].value_counts(normalize=True).sort_index())


# ============================================================
# 6. Generate potential outcomes and known individual CATE
# ============================================================

def create_potential_outcomes(
    X: pd.DataFrame,
    random_generator: np.random.Generator,
) -> pd.DataFrame:
    baseline_mean = linear_score(
        X,
        BASELINE_INTERCEPT,
        BASELINE_COEFFICIENTS,
    )

    true_tau = linear_score(
        X,
        TAU_INTERCEPT,
        TAU_COEFFICIENTS,
    )

    noise = random_generator.normal(
        loc=0.0,
        scale=1.0,
        size=len(X),
    )

    y0 = baseline_mean + noise
    y1 = y0 + true_tau

    return pd.DataFrame(
        {
            "Y0": y0,
            "Y1": y1,
            "true_tau": true_tau,
        },
        index=X.index,
    )


train_outcomes = create_potential_outcomes(
    X_train_model,
    rng,
)

test_outcomes = create_potential_outcomes(
    X_test_model,
    rng,
)

train_df = pd.concat(
    [train_df, train_outcomes],
    axis=1,
)

test_df = pd.concat(
    [test_df, test_outcomes],
    axis=1,
)

train_df["Y"] = (
    train_df["A"] * train_df["Y1"]
    + (1 - train_df["A"]) * train_df["Y0"]
)

test_df["Y"] = (
    test_df["A"] * test_df["Y1"]
    + (1 - test_df["A"]) * test_df["Y0"]
)

print("\nTrain true CATE summary:")
print(train_df["true_tau"].describe())

print("\nTest true CATE summary:")
print(test_df["true_tau"].describe())


# ============================================================
# 7. Sanity checks
# ============================================================

true_ate_train = train_df["true_tau"].mean()

naive_difference_train = (
    train_df.loc[train_df["A"] == 1, "Y"].mean()
    - train_df.loc[train_df["A"] == 0, "Y"].mean()
)

print("\nTrue train ATE:", true_ate_train)
print("Naive treated-control difference:", naive_difference_train)
print("Naive bias:", naive_difference_train - true_ate_train)

if not (0 < train_df["A"].mean() < 1):
    raise ValueError(
        "Treatment assignment contains only one group in train."
    )

if not (0 < test_df["A"].mean() < 1):
    raise ValueError(
        "Treatment assignment contains only one group in test."
    )

if not np.isfinite(
    train_df[["Y0", "Y1", "Y", "true_tau"]].to_numpy()
).all():
    raise ValueError(
        "Train outcomes contain NaN, inf, or -inf."
    )

if not np.isfinite(
    test_df[["Y0", "Y1", "Y", "true_tau"]].to_numpy()
).all():
    raise ValueError(
        "Test outcomes contain NaN, inf, or -inf."
    )

# Check every encoded variable used by tau(X) has non-zero variance.
tau_feature_variances = X_train_model[
    list(TAU_COEFFICIENTS)
].var()

zero_variance_tau_features = tau_feature_variances[
    tau_feature_variances == 0
].index.tolist()

if zero_variance_tau_features:
    raise ValueError(
        "The following CATE features have zero training variance: "
        f"{zero_variance_tau_features}"
    )

print("\nAll CATE mechanism features have non-zero training variance.")


# ============================================================
# 8. Save a transparent DGP specification
# ============================================================

dgp_rows = []

for mechanism_name, intercept, coefficient_dict in [
    (
        "propensity_score",
        PROPENSITY_INTERCEPT,
        PROPENSITY_COEFFICIENTS,
    ),
    (
        "baseline_outcome_y0",
        BASELINE_INTERCEPT,
        BASELINE_COEFFICIENTS,
    ),
    (
        "true_tau",
        TAU_INTERCEPT,
        TAU_COEFFICIENTS,
    ),
]:
    dgp_rows.append(
        {
            "mechanism": mechanism_name,
            "feature": "INTERCEPT",
            "coefficient": intercept,
        }
    )

    for feature_name, coefficient in coefficient_dict.items():
        dgp_rows.append(
            {
                "mechanism": mechanism_name,
                "feature": feature_name,
                "coefficient": coefficient,
            }
        )

dgp_spec_df = pd.DataFrame(dgp_rows)


# ============================================================
# 9. Save fixed data artifacts
# ============================================================

print("\nTrain dataset shape:", train_df.shape)
print("Test dataset shape:", test_df.shape)

safe_to_csv(
    train_df,
    TRAIN_OUTPUT_PATH,
    OVERWRITE_EXISTING,
)

safe_to_csv(
    test_df,
    TEST_OUTPUT_PATH,
    OVERWRITE_EXISTING,
)

safe_to_csv(
    dgp_spec_df,
    DGP_SPEC_OUTPUT_PATH,
    OVERWRITE_EXISTING,
)

print("\nOutput files:")
print(TRAIN_OUTPUT_PATH)
print(TEST_OUTPUT_PATH)
print(DGP_SPEC_OUTPUT_PATH)
