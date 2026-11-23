import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from pipeline_utils import (
    find_data_root,
    get_run_paths,
    load_experiment_config,
    snapshot_metadata,
    validate_condition_config,
    validate_patient_ids,
)


# ============================================================
# PATH CONFIGURATION
# ------------------------------------------------------------
# This script reads declared text files and creates BERT CLS
# embeddings for the active experiment config.
# ============================================================

DATA_ROOT = find_data_root()
CONFIG, CONFIG_PATH = load_experiment_config(DATA_ROOT)
validate_condition_config(CONFIG)
RUN_PATHS = get_run_paths(DATA_ROOT, CONFIG)

model_name = "bert-base-uncased"
batch_size = 16
OVERWRITE_EXISTING = False


# ============================================================
# 1. Load tokenizer and BERT model
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("Using device:", device)
print("Loaded model:", model_name)
print("Embedding size:", model.config.hidden_size)


# ============================================================
# 2. Create CLS embeddings
# ============================================================

def create_cls_embeddings(texts):
    all_embeddings = []

    for start_index in range(0, len(texts), batch_size):
        batch_texts = texts[start_index:start_index + batch_size]

        encoded_batch = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        encoded_batch = {
            key: value.to(device)
            for key, value in encoded_batch.items()
        }

        with torch.no_grad():
            model_outputs = model(**encoded_batch)

        cls_embeddings = model_outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.cpu())

        print(
            "Processed texts "
            f"{start_index + 1}-{min(start_index + batch_size, len(texts))} "
            f"out of {len(texts)}"
        )

    return torch.cat(all_embeddings, dim=0).numpy()


# ============================================================
# 3. Create embeddings for declared conditions
# ============================================================

snapshot_metadata(DATA_ROOT, CONFIG_PATH, CONFIG)

for condition in CONFIG["conditions"]:
    text_path = RUN_PATHS["text_dir"] / condition["text_filename"]
    embedding_path = (
        RUN_PATHS["embeddings_dir"]
        / condition["embedding_filename"]
    )

    if embedding_path.exists() and not OVERWRITE_EXISTING:
        print(
            "[skip ] exists, not overwriting "
            f"(set OVERWRITE_EXISTING=True): {embedding_path}"
        )
        continue

    if not text_path.exists():
        raise FileNotFoundError(f"Missing text file: {text_path}")

    text_df = pd.read_csv(text_path)

    if "text" not in text_df.columns:
        raise ValueError(f"Text file has no text column: {text_path}")

    validate_patient_ids(text_df, f"Text file {text_path.name}")

    if text_df["text"].isna().any() or (text_df["text"].str.strip() == "").any():
        raise ValueError(f"Text file contains missing or empty text: {text_path}")

    print("\nProcessing:", text_path)
    print("Number of texts:", len(text_df))

    embeddings = create_cls_embeddings(text_df["text"].tolist())

    if embeddings.shape[1] != model.config.hidden_size:
        raise ValueError("Unexpected embedding dimension.")

    embedding_columns = [
        f"emb_{index}"
        for index in range(embeddings.shape[1])
    ]

    embeddings_df = pd.DataFrame(embeddings, columns=embedding_columns)
    embeddings_df.insert(0, "patient_id", text_df["patient_id"].values)

    validate_patient_ids(embeddings_df, f"Embeddings for {condition['name']}")

    if len(embeddings_df) != len(text_df):
        raise ValueError("Embedding row count does not match text row count.")

    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings_df.to_csv(embedding_path, index=False)

    print("[write]", embedding_path)
    print("Embedding matrix shape:", embeddings.shape)

print("\nFinished creating embeddings.")
