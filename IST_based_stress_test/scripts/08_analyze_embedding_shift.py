import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

from pipeline_utils import (
    find_data_root,
    get_evaluation_conditions,
    get_reference_condition,
    get_run_paths,
    load_experiment_config,
    snapshot_metadata,
    validate_condition_config,
    safe_to_csv,
)


# ============================================================
# PATH CONFIGURATION
# ------------------------------------------------------------
# This script compares test embeddings with the declared
# reference condition and links shift to prediction error.
# ============================================================

DATA_ROOT = find_data_root()
CONFIG, CONFIG_PATH = load_experiment_config(DATA_ROOT)
validate_condition_config(CONFIG)
RUN_PATHS = get_run_paths(DATA_ROOT, CONFIG)

OVERWRITE_EXISTING = False

REFERENCE_CONFIG = get_reference_condition(CONFIG)
REFERENCE_CONDITION = REFERENCE_CONFIG["name"]
REFERENCE_EMBEDDINGS_PATH = (
    RUN_PATHS["embeddings_dir"]
    / REFERENCE_CONFIG["embedding_filename"]
)

COMPARISON_CONFIGS = [
    condition
    for condition in get_evaluation_conditions(CONFIG)
    if condition["name"] != REFERENCE_CONDITION
]

COMPARISON_EMBEDDINGS_PATHS = {
    condition["name"]: (
        RUN_PATHS["embeddings_dir"]
        / condition["embedding_filename"]
    )
    for condition in COMPARISON_CONFIGS
}

PREDICTIONS_PATH = RUN_PATHS["results_dir"] / "text_condition_predictions.csv"
RESULTS_PATH = RUN_PATHS["results_dir"] / "embedding_shift_results.csv"

snapshot_metadata(DATA_ROOT, CONFIG_PATH, CONFIG)

print("Experiment:", CONFIG["experiment_id"])
print("Reference condition:", REFERENCE_CONDITION)
print("Comparisons:", list(COMPARISON_EMBEDDINGS_PATHS))

# ============================================================
# VALIDATION HELPERS
# ============================================================

def load_embedding_file(
    file_path: Path,
    label: str,
) -> tuple[pd.DataFrame, list[str]]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing {label} embedding file: {file_path}"
        )

    data_frame = pd.read_csv(file_path)

    if "patient_id" not in data_frame.columns:
        raise ValueError(
            f"{label} embeddings do not contain patient_id."
        )

    embedding_columns = [
        column
        for column in data_frame.columns
        if column.startswith("emb_")
    ]

    if not embedding_columns:
        raise ValueError(
            f"{label} embeddings contain no emb_ columns."
        )

    if data_frame["patient_id"].duplicated().any():
        raise ValueError(
            f"{label} embeddings contain duplicate patient IDs."
        )

    if data_frame[embedding_columns].isna().any().any():
        raise ValueError(
            f"{label} embeddings contain missing values."
        )

    values = data_frame[embedding_columns].to_numpy()

    if not np.isfinite(values).all():
        raise ValueError(
            f"{label} embeddings contain inf or -inf values."
        )

    return data_frame, embedding_columns


def validate_matching_ids(
    reference_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    comparison_name: str,
) -> None:
    reference_ids = set(reference_df["patient_id"])
    comparison_ids = set(comparison_df["patient_id"])

    if reference_ids != comparison_ids:
        only_reference = sorted(reference_ids - comparison_ids)[:10]
        only_comparison = sorted(comparison_ids - reference_ids)[:10]

        raise ValueError(
            f"Patient IDs do not match for {comparison_name}. "
            f"Only in reference (first 10): {only_reference}; "
            f"only in comparison (first 10): {only_comparison}."
        )


# ============================================================
# DISTANCE CALCULATIONS
# ============================================================

def calculate_embedding_distances(
    reference_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    embedding_columns: list[str],
    comparison_name: str,
) -> pd.DataFrame:
    validate_matching_ids(
        reference_df,
        comparison_df,
        comparison_name,
    )

    reference_renamed = reference_df.rename(
        columns={
            column: f"{column}_reference"
            for column in embedding_columns
        }
    )

    comparison_renamed = comparison_df.rename(
        columns={
            column: f"{column}_{comparison_name}"
            for column in embedding_columns
        }
    )

    merged_df = reference_renamed.merge(
        comparison_renamed,
        on="patient_id",
        how="inner",
        validate="one_to_one",
    )

    expected_rows = len(reference_df)

    if len(merged_df) != expected_rows:
        raise ValueError(
            f"Merged row count for {comparison_name} is "
            f"{len(merged_df)}, expected {expected_rows}."
        )

    reference_columns = [
        f"{column}_reference"
        for column in embedding_columns
    ]

    comparison_columns = [
        f"{column}_{comparison_name}"
        for column in embedding_columns
    ]

    reference_matrix = merged_df[reference_columns].to_numpy()
    comparison_matrix = merged_df[comparison_columns].to_numpy()

    row_cosine_similarity = np.diag(
        cosine_similarity(
            reference_matrix,
            comparison_matrix,
        )
    )

    cosine_distance = 1.0 - row_cosine_similarity

    euclidean_distance = np.linalg.norm(
        reference_matrix - comparison_matrix,
        axis=1,
    )

    return pd.DataFrame(
        {
            "patient_id": merged_df["patient_id"],
            "comparison": comparison_name,
            "cosine_similarity": row_cosine_similarity,
            "cosine_distance": cosine_distance,
            "euclidean_distance": euclidean_distance,
        }
    )


# ============================================================
# LOAD REFERENCE EMBEDDINGS
# ============================================================

reference_embeddings, embedding_columns = load_embedding_file(
    REFERENCE_EMBEDDINGS_PATH,
    label=REFERENCE_CONDITION,
)

print("Number of embedding dimensions:", len(embedding_columns))
print(
    f"{REFERENCE_CONDITION} embedding shape:",
    reference_embeddings.shape,
)


# ============================================================
# COMPUTE EMBEDDING SHIFTS
# ============================================================

all_shift_frames = []

for comparison_name, comparison_path in (
    COMPARISON_EMBEDDINGS_PATHS.items()
):
    comparison_embeddings, comparison_embedding_columns = (
        load_embedding_file(
            comparison_path,
            label=comparison_name,
        )
    )

    if comparison_embedding_columns != embedding_columns:
        raise ValueError(
            f"Embedding columns differ between "
            f"{REFERENCE_CONDITION} and {comparison_name}."
        )

    shift_df = calculate_embedding_distances(
        reference_df=reference_embeddings,
        comparison_df=comparison_embeddings,
        embedding_columns=embedding_columns,
        comparison_name=comparison_name,
    )

    print(
        f"\n{REFERENCE_CONDITION}-{comparison_name} "
        f"merged shape: ({len(shift_df)}, "
        f"{1 + 2 * len(embedding_columns)})"
    )

    all_shift_frames.append(shift_df)

embedding_shift_patient_level = pd.concat(
    all_shift_frames,
    ignore_index=True,
)


# ============================================================
# SUMMARY TABLE
# ============================================================

embedding_shift_summary = (
    embedding_shift_patient_level
    .groupby("comparison", as_index=False)
    .agg(
        n_patients=("patient_id", "count"),
        mean_cosine_similarity=(
            "cosine_similarity",
            "mean",
        ),
        std_cosine_similarity=(
            "cosine_similarity",
            "std",
        ),
        mean_cosine_distance=(
            "cosine_distance",
            "mean",
        ),
        mean_euclidean_distance=(
            "euclidean_distance",
            "mean",
        ),
    )
    .sort_values(
        "comparison",
    )
    .reset_index(drop=True)
)

print("\n" + "=" * 60)
print("Embedding-shift summary")
print("=" * 60)
print(embedding_shift_summary)


# ============================================================
# PREDICTION-ERROR DEGRADATION ANALYSIS
# ============================================================

if not PREDICTIONS_PATH.exists():
    raise FileNotFoundError(
        f"Missing prediction file: {PREDICTIONS_PATH}"
    )

predictions_df = pd.read_csv(PREDICTIONS_PATH)

required_prediction_columns = {
    "patient_id",
    "condition",
    "true_tau",
    "predicted_cate",
}

missing_prediction_columns = (
    required_prediction_columns
    - set(predictions_df.columns)
)

if missing_prediction_columns:
    raise ValueError(
        "Predictions file is missing required columns: "
        f"{sorted(missing_prediction_columns)}"
    )

if predictions_df.duplicated(
    subset=["patient_id", "condition"]
).any():
    raise ValueError(
        "Predictions contain duplicate patient_id-condition rows."
    )

full_predictions = predictions_df.loc[
    predictions_df["condition"] == REFERENCE_CONDITION,
    [
        "patient_id",
        "true_tau",
        "predicted_cate",
    ],
].copy()

if full_predictions.empty:
    raise ValueError(
        f"No predictions with condition == "
        f"'{REFERENCE_CONDITION}' were found."
    )

full_predictions = full_predictions.rename(
    columns={
        "predicted_cate": "predicted_cate_full",
    }
)

full_predictions["absolute_error_full"] = np.abs(
    full_predictions["predicted_cate_full"]
    - full_predictions["true_tau"]
)

for comparison_name in COMPARISON_EMBEDDINGS_PATHS:
    comparison_predictions = predictions_df.loc[
        predictions_df["condition"] == comparison_name,
        [
            "patient_id",
            "true_tau",
            "predicted_cate",
        ],
    ].copy()

    if comparison_predictions.empty:
        print(
            f"\n[warn ] no predictions with condition == "
            f"'{comparison_name}', skipping degradation analysis."
        )
        continue

    comparison_predictions = comparison_predictions.rename(
        columns={
            "predicted_cate": "predicted_cate_comparison",
            "true_tau": "true_tau_comparison",
        }
    )

    comparison_predictions["absolute_error_comparison"] = np.abs(
        comparison_predictions["predicted_cate_comparison"]
        - comparison_predictions["true_tau_comparison"]
    )

    condition_shift_df = embedding_shift_patient_level.loc[
        embedding_shift_patient_level["comparison"]
        == comparison_name
    ].copy()

    degradation_df = (
        condition_shift_df
        .merge(
            full_predictions,
            on="patient_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            comparison_predictions,
            on="patient_id",
            how="inner",
            validate="one_to_one",
        )
    )

    if len(degradation_df) != len(full_predictions):
        raise ValueError(
            f"Prediction merge for {comparison_name} retained "
            f"{len(degradation_df)} rows; expected "
            f"{len(full_predictions)}."
        )

    if not np.allclose(
        degradation_df["true_tau"],
        degradation_df["true_tau_comparison"],
    ):
        raise ValueError(
            f"true_tau differs between full and {comparison_name}."
        )

    degradation_df["error_increase"] = (
        degradation_df["absolute_error_comparison"]
        - degradation_df["absolute_error_full"]
    )

    correlation = degradation_df[
        ["cosine_distance", "error_increase"]
    ].corr().iloc[0, 1]

    print("\n" + "=" * 60)
    print(
        "Embedding shift versus error increase:",
        comparison_name,
    )
    print("=" * 60)
    print(
        "Correlation between cosine distance and error increase:"
    )
    print(correlation)

    # Prediction quintiles for diagnostic inspection.
    degradation_df["prediction_group"] = pd.qcut(
        degradation_df["predicted_cate_comparison"],
        q=5,
        duplicates="drop",
    )

    quintile_summary = (
        degradation_df
        .groupby(
            "prediction_group",
            observed=False,
        )
        .agg(
            n_patients=("patient_id", "count"),
            mean_predicted_cate=(
                "predicted_cate_comparison",
                "mean",
            ),
            mean_true_cate=("true_tau", "mean"),
            mean_absolute_error=(
                "absolute_error_comparison",
                "mean",
            ),
            min_prediction=(
                "predicted_cate_comparison",
                "min",
            ),
            max_prediction=(
                "predicted_cate_comparison",
                "max",
            ),
        )
    )

    print("\n" + "=" * 60)
    print(
        f"{comparison_name}: predictions by predicted-CATE "
        "quintile"
    )
    print("=" * 60)
    print(quintile_summary)

    highest_predictions = (
        degradation_df[
            [
                "patient_id",
                "true_tau",
                "predicted_cate_comparison",
                "absolute_error_comparison",
            ]
        ]
        .rename(
            columns={
                "predicted_cate_comparison": "predicted_cate",
                "absolute_error_comparison": "absolute_error",
            }
        )
        .sort_values(
            "predicted_cate",
            ascending=False,
        )
        .head(20)
    )

    print("\n" + "=" * 60)
    print(f"{comparison_name}: twenty highest predictions")
    print("=" * 60)
    print(highest_predictions.to_string(index=False))


# ============================================================
# SAVE SUMMARY
# ============================================================

safe_to_csv(
    data_frame=embedding_shift_summary,
    output_path=RESULTS_PATH,
    overwrite=OVERWRITE_EXISTING,
)

print("\nTarget embedding-shift results:")
print(RESULTS_PATH)
