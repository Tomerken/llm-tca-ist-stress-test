import numpy as np
import pandas as pd

from pipeline_utils import (
    find_data_root,
    get_run_paths,
    load_experiment_config,
    validate_condition_config,
    validate_patient_ids,
)


# ============================================================
# PATH CONFIGURATION
# ------------------------------------------------------------
# Console-only validation. No files are written here.
# ============================================================

DATA_ROOT = find_data_root()
CONFIG, _ = load_experiment_config(DATA_ROOT)
validate_condition_config(CONFIG)
RUN_PATHS = get_run_paths(DATA_ROOT, CONFIG)

TRAIN_INPUT_PATH = DATA_ROOT / CONFIG["fixed_inputs"]["train_text_input"]
TEST_INPUT_PATH = DATA_ROOT / CONFIG["fixed_inputs"]["test_text_input"]
EXPECTED_EMBEDDING_DIMENSION = 768


# ============================================================
# 1. Validate declared embedding files
# ============================================================

for condition in CONFIG["conditions"]:
    split = condition["split"]
    input_path = TRAIN_INPUT_PATH if split == "train" else TEST_INPUT_PATH
    embedding_path = (
        RUN_PATHS["embeddings_dir"]
        / condition["embedding_filename"]
    )

    print("\n" + "=" * 60)
    print("Condition:", condition["name"])
    print("Split:", split)

    if not embedding_path.exists():
        print("[skip ] file not found:", embedding_path)
        continue

    input_df = pd.read_csv(input_path)
    embedding_df = pd.read_csv(embedding_path)

    validate_patient_ids(input_df, f"{split} text input")
    validate_patient_ids(embedding_df, f"{condition['name']} embeddings")

    embedding_columns = [
        column
        for column in embedding_df.columns
        if column.startswith("emb_")
    ]

    if len(embedding_columns) != EXPECTED_EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected {EXPECTED_EMBEDDING_DIMENSION} embedding columns, "
            f"found {len(embedding_columns)} in {embedding_path}."
        )

    if embedding_df[embedding_columns].isna().any().any():
        raise ValueError(f"Missing embedding values in {embedding_path}.")

    embedding_values = embedding_df[embedding_columns].to_numpy()
    if not np.isfinite(embedding_values).all():
        raise ValueError(f"Infinite embedding values in {embedding_path}.")

    if len(embedding_df) != len(input_df):
        raise ValueError(
            f"Embedding row count differs from {split} input: "
            f"{len(embedding_df)} vs {len(input_df)}."
        )

    if set(embedding_df["patient_id"]) != set(input_df["patient_id"]):
        raise ValueError(f"Patient IDs differ for {embedding_path}.")

    print("Path:", embedding_path)
    print("Shape:", embedding_df.shape)
    print("Embedding columns:", len(embedding_columns))
    print("Duplicate patient IDs:", embedding_df["patient_id"].duplicated().sum())
    print("Missing values:", embedding_df.isna().sum().sum())
    print("Infinite values:", np.isinf(embedding_values).sum())
    print("First patient ID:", embedding_df["patient_id"].iloc[0])
    print("Last patient ID:", embedding_df["patient_id"].iloc[-1])

print("\nFinished embedding validation.")
