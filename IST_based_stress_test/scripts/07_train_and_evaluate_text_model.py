import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from pipeline_utils import (
    find_data_root,
    get_evaluation_conditions,
    get_run_paths,
    get_training_condition,
    load_experiment_config,
    snapshot_metadata,
    validate_condition_config,
    validate_patient_ids,
    safe_to_csv,
)


# ============================================================
# PATH CONFIGURATION
# ------------------------------------------------------------
# The model architecture, hyperparameters, scaling, and training
# loop are unchanged. This script only reads the active config.
# ============================================================

DATA_ROOT = find_data_root()
CONFIG, CONFIG_PATH = load_experiment_config(DATA_ROOT)
validate_condition_config(CONFIG)
RUN_PATHS = get_run_paths(DATA_ROOT, CONFIG)

OVERWRITE_EXISTING = False

PSEUDO_OUTCOMES_PATH = DATA_ROOT / CONFIG["fixed_inputs"]["pseudo_outcomes"]
TEST_TRUE_TAU_PATH = DATA_ROOT / CONFIG["fixed_inputs"]["test_causal_data"]

TRAINING_CONDITION = get_training_condition(CONFIG)
EVALUATION_CONDITIONS = get_evaluation_conditions(CONFIG)

TRAIN_EMBEDDINGS_PATH = (
    RUN_PATHS["embeddings_dir"]
    / TRAINING_CONDITION["embedding_filename"]
)

test_embedding_files = {
    condition["name"]: (
        RUN_PATHS["embeddings_dir"]
        / condition["embedding_filename"]
    )
    for condition in EVALUATION_CONDITIONS
}

RESULTS_OUTPUT_PATH = RUN_PATHS["results_dir"] / "text_condition_results.csv"
PREDICTIONS_OUTPUT_PATH = RUN_PATHS["results_dir"] / "text_condition_predictions.csv"

# Training hyperparameters are unchanged.
batch_size = 128
learning_rate = 0.00005
num_epochs = 100
random_seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(random_seed)
np.random.seed(random_seed)

snapshot_metadata(DATA_ROOT, CONFIG_PATH, CONFIG)

print("Using device:", device)
print("Experiment:", CONFIG["experiment_id"])
print("Training condition:", TRAINING_CONDITION["name"])
print("Evaluating conditions:", list(test_embedding_files))

# ============================================================
# 2. Load and merge train embeddings with pseudo-outcomes
# ============================================================

train_embeddings_df = pd.read_csv(TRAIN_EMBEDDINGS_PATH)
pseudo_outcomes_df = pd.read_csv(PSEUDO_OUTCOMES_PATH)
validate_patient_ids(train_embeddings_df, "Train embeddings")
validate_patient_ids(pseudo_outcomes_df, "Pseudo-outcomes")

train_df = train_embeddings_df.merge(
    pseudo_outcomes_df,
    on="patient_id",
    how="inner",
    validate="one_to_one",
)

print("\nTrain embeddings shape:", train_embeddings_df.shape)
print("Pseudo-outcomes shape:", pseudo_outcomes_df.shape)
print("Merged training data shape:", train_df.shape)

if len(train_df) != len(train_embeddings_df) or len(train_df) != len(pseudo_outcomes_df):
    raise ValueError("Training merge did not retain all patient IDs.")

print("\nMissing pseudo-outcomes after merge:")
print(train_df["pseudo_outcome"].isna().sum())

print("\nFirst five merged rows:")
print(train_df[["patient_id", "pseudo_outcome"]].head())

# ============================================================
# 3. Prepare train features and targets
# ============================================================

embedding_columns = [
    column
    for column in train_df.columns
    if column.startswith("emb_")
]

X_train = train_df[embedding_columns].to_numpy()
y_train = train_df["pseudo_outcome"].to_numpy()

print("\nX_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("\nFirst five pseudo-outcomes:")
print(y_train[:5])

# ============================================================
# 4. Normalize train embeddings and pseudo-outcomes
# ============================================================

# The scaler is fitted only on the training embeddings.
# Later, the same train statistics will be applied to every test condition.
embedding_scaler = StandardScaler()

X_train_scaled = embedding_scaler.fit_transform(X_train)

# Normalizing the pseudo-outcome helps the neural network train more stably.
# We will inverse-transform predictions before evaluating PEHE.
target_mean = y_train.mean()
target_std = y_train.std()

y_train_scaled = (y_train - target_mean) / target_std

print("\nScaled X_train mean:", X_train_scaled.mean())
print("Scaled X_train standard deviation:", X_train_scaled.std())

print("\nPseudo-outcome mean before scaling:", target_mean)
print("Pseudo-outcome standard deviation before scaling:", target_std)

# ============================================================
# 5. Convert training data to PyTorch tensors
# ============================================================

X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32,
)

y_train_tensor = torch.tensor(
    y_train_scaled.reshape(-1, 1),
    dtype=torch.float32,
)

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

print("\nNumber of training batches:", len(train_loader))

# ============================================================
# 6. Define the text-based CATE regression network
# ============================================================

class RegressionNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims=[256, 128, 64],
        dropout_rate=0.3,
    ):
        super().__init__()

        layers = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout_rate),
                ]
            )

            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ============================================================
# 7. Initialize model, optimizer, and loss function
# ============================================================

input_dim = X_train_scaled.shape[1]

model = RegressionNet(
    input_dim=input_dim,
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
)

criterion = nn.MSELoss()

print("\nModel input dimension:", input_dim)
print(model)

# ============================================================
# 8. Train the text model on full-text embeddings only
# ============================================================

for epoch in range(num_epochs):
    model.train()

    total_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        predictions = model(X_batch)

        loss = criterion(
            predictions,
            y_batch,
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Training loss: {average_loss:.4f}"
        )

# ============================================================
# 9. Load test data and true CATE values
# ============================================================

test_data_df = pd.read_csv(TEST_TRUE_TAU_PATH)
validate_patient_ids(test_data_df, "Test causal data")

true_cate_df = test_data_df[
    [
        "patient_id",
        "true_tau",
    ]
].copy()

print("\nTest data shape:", test_data_df.shape)
print("True CATE rows:", len(true_cate_df))

print("\nFirst five true CATE values:")
print(true_cate_df.head())

# ============================================================
# 10. Predict CATE values for one embedding file
# ============================================================

def predict_cate_for_condition(
    embeddings_path,
    condition_name,
):
    embeddings_df = pd.read_csv(embeddings_path)
    validate_patient_ids(embeddings_df, f"{condition_name} embeddings")

    condition_df = embeddings_df.merge(
        true_cate_df,
        on="patient_id",
        how="inner",
        validate="one_to_one",
    )

    if len(condition_df) != len(true_cate_df):
        raise ValueError(f"Test merge did not retain all patient IDs for {condition_name}.")

    X_condition = condition_df[embedding_columns].to_numpy()

    # Use only training-set scaling statistics.
    X_condition_scaled = embedding_scaler.transform(
        X_condition
    )

    X_condition_tensor = torch.tensor(
        X_condition_scaled,
        dtype=torch.float32,
    ).to(device)

    model.eval()

    with torch.no_grad():
        predicted_cate_scaled = model(
            X_condition_tensor
        ).cpu().numpy().reshape(-1)

    # Convert predictions back to the original pseudo-outcome scale.
    predicted_cate = (
        predicted_cate_scaled * target_std
        + target_mean
    )

    true_cate = condition_df["true_tau"].to_numpy()

    prediction_df = pd.DataFrame(
        {
            "patient_id": condition_df["patient_id"],
            "true_tau": true_cate,
            "predicted_cate": predicted_cate,
            "condition": condition_name,
        }
    )

    return prediction_df

# ============================================================
# 11. Evaluate the configured test conditions
# ============================================================

all_prediction_dfs = []
results = []

for condition_name, embeddings_path in test_embedding_files.items():
    if not embeddings_path.exists():
        print(
            f"[warn ] embeddings missing, skipping condition "
            f"'{condition_name}': {embeddings_path}"
        )
        continue

    prediction_df = predict_cate_for_condition(
        embeddings_path=embeddings_path,
        condition_name=condition_name,
    )

    prediction_error = (
        prediction_df["predicted_cate"]
        - prediction_df["true_tau"]
    )

    mse = np.mean(prediction_error ** 2)
    mae = np.mean(np.abs(prediction_error))
    pehe = np.sqrt(mse)

    results.append(
        {
            "condition": condition_name,
            "n_patients": len(prediction_df),
            "mean_predicted_cate": prediction_df[
                "predicted_cate"
            ].mean(),
            "mean_true_cate": prediction_df[
                "true_tau"
            ].mean(),
            "mse": mse,
            "mae": mae,
            "pehe": pehe,
        }
    )

    all_prediction_dfs.append(prediction_df)

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("CATE estimation results")
print("=" * 60)

print(
    results_df[
        [
            "condition",
            "n_patients",
            "mean_predicted_cate",
            "mean_true_cate",
            "mse",
            "mae",
            "pehe",
        ]
    ]
)

# ============================================================
# 12. Save predictions and summary results
# ============================================================

all_predictions_df = pd.concat(
    all_prediction_dfs,
    ignore_index=True,
)

safe_to_csv(results_df, RESULTS_OUTPUT_PATH, OVERWRITE_EXISTING)
safe_to_csv(all_predictions_df, PREDICTIONS_OUTPUT_PATH, OVERWRITE_EXISTING)

print("\nTarget result summary:")
print(RESULTS_OUTPUT_PATH)

print("\nTarget patient-level predictions:")
print(PREDICTIONS_OUTPUT_PATH)

# ============================================================
# 13. Inspect prediction distributions by condition
# ============================================================

prediction_summary = (
    all_predictions_df
    .groupby("condition")["predicted_cate"]
    .agg(
        count="count",
        mean="mean",
        std="std",
        min="min",
        q25=lambda values: values.quantile(0.25),
        median="median",
        q75=lambda values: values.quantile(0.75),
        max="max",
    )
)

print("\nPrediction distribution by condition:")
print(prediction_summary)

true_cate_summary = (
    all_predictions_df
    .groupby("condition")["true_tau"]
    .agg(
        mean="mean",
        std="std",
        min="min",
        median="median",
        max="max",
    )
)

print("\nTrue CATE distribution:")
print(true_cate_summary)
