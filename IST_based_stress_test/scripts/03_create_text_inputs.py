import pandas as pd

from pipeline_utils import (
    find_data_root,
    safe_to_csv,
    validate_patient_ids,
)


# ============================================================
# PATH CONFIGURATION
# ------------------------------------------------------------
# This script creates fixed clean tables for text generation.
# They contain only text-visible pre-treatment covariates.
# ============================================================

DATA_ROOT = find_data_root()
SEMISYN_DIR = DATA_ROOT / "01_semisynthetic_data"
TEXT_INPUT_DIR = DATA_ROOT / "03_text_inputs"

TRAIN_DATA_PATH = SEMISYN_DIR / "IST_syn_train.csv"
TEST_DATA_PATH = SEMISYN_DIR / "IST_syn_test.csv"

TRAIN_OUTPUT_PATH = TEXT_INPUT_DIR / "IST_train_text_input.csv"
TEST_OUTPUT_PATH = TEXT_INPUT_DIR / "IST_test_text_input.csv"

OVERWRITE_EXISTING = False


# ============================================================
# 1. Load fixed semi-synthetic data
# ============================================================

train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)

validate_patient_ids(train_df, "Training data")
validate_patient_ids(test_df, "Test data")

print("Train dataset shape:", train_df.shape)
print("Test dataset shape:", test_df.shape)


# ============================================================
# 2. Keep only allowed text variables
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

text_input_columns = ["patient_id"] + feature_columns

for label, data_frame in {
    "Training data": train_df,
    "Test data": test_df,
}.items():
    missing_columns = [
        column
        for column in text_input_columns
        if column not in data_frame.columns
    ]
    if missing_columns:
        raise ValueError(
            f"{label} is missing columns: {missing_columns}"
        )

train_text_input = train_df[text_input_columns].copy()
test_text_input = test_df[text_input_columns].copy()

# These columns must never enter text generation.
for forbidden_column in ["A", "Y", "Y0", "Y1", "true_tau", "pseudo_outcome"]:
    if forbidden_column in train_text_input.columns:
        raise ValueError(f"Forbidden column in train text input: {forbidden_column}")
    if forbidden_column in test_text_input.columns:
        raise ValueError(f"Forbidden column in test text input: {forbidden_column}")

validate_patient_ids(train_text_input, "Train text input")
validate_patient_ids(test_text_input, "Test text input")


# ============================================================
# 3. Save fixed text-input tables
# ============================================================

safe_to_csv(
    train_text_input,
    TRAIN_OUTPUT_PATH,
    OVERWRITE_EXISTING,
)
safe_to_csv(
    test_text_input,
    TEST_OUTPUT_PATH,
    OVERWRITE_EXISTING,
)

print("\nTarget clean text-input files:")
print(TRAIN_OUTPUT_PATH)
print(TEST_OUTPUT_PATH)

print("\nFirst five training rows:")
print(train_text_input.head())
