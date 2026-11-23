import json
import shutil
from pathlib import Path

import pandas as pd


def find_data_root(marker: str = "IST_based_stress_test") -> Path:
    """Find IST_based_stress_test from the script location or IST_DATA_ROOT."""
    import os

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
        "Set IST_DATA_ROOT to point at it."
    )


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


def load_experiment_config(
    data_root: Path,
    config_name: str = "experiment_config.json",
) -> tuple[dict, Path]:
    config_path = data_root / "04_protocols" / config_name

    if not config_path.exists():
        raise FileNotFoundError(f"Missing experiment config: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    required_keys = {
        "experiment_id",
        "run_dir",
        "fixed_inputs",
        "conditions",
        "outputs",
    }
    missing_keys = required_keys - set(config)

    if missing_keys:
        raise ValueError(
            "Experiment config is missing keys: "
            f"{sorted(missing_keys)}"
        )

    if not isinstance(config["conditions"], list) or not config["conditions"]:
        raise ValueError("Experiment config must contain a non-empty conditions list.")

    return config, config_path


def get_run_paths(data_root: Path, config: dict) -> dict[str, Path]:
    run_dir = data_root / config["run_dir"]
    outputs = config["outputs"]

    required_output_keys = {
        "text_dir",
        "embeddings_dir",
        "results_dir",
        "metadata_dir",
    }
    missing_output_keys = required_output_keys - set(outputs)

    if missing_output_keys:
        raise ValueError(
            "Config outputs are missing keys: "
            f"{sorted(missing_output_keys)}"
        )

    return {
        "run_dir": run_dir,
        "text_dir": run_dir / outputs["text_dir"],
        "embeddings_dir": run_dir / outputs["embeddings_dir"],
        "results_dir": run_dir / outputs["results_dir"],
        "metadata_dir": run_dir / outputs["metadata_dir"],
    }


def validate_condition_config(config: dict) -> None:
    required_condition_keys = {
        "name",
        "split",
        "generator",
        "text_filename",
        "embedding_filename",
        "train_model",
        "evaluate",
        "is_reference",
    }

    seen_pairs = set()
    train_rows = []
    reference_rows = []

    for condition in config["conditions"]:
        missing_keys = required_condition_keys - set(condition)
        if missing_keys:
            raise ValueError(
                "A condition is missing keys: "
                f"{sorted(missing_keys)}"
            )

        pair = (condition["name"], condition["split"])
        if pair in seen_pairs:
            raise ValueError(
                "Duplicate condition/split pair in config: "
                f"{pair}"
            )
        seen_pairs.add(pair)

        if condition["split"] not in {"train", "test"}:
            raise ValueError(
                "Condition split must be 'train' or 'test': "
                f"{condition['split']!r}"
            )

        if condition["train_model"]:
            train_rows.append(condition)

        if condition["is_reference"]:
            reference_rows.append(condition)

    if len(train_rows) != 1:
        raise ValueError(
            "Exactly one condition must have train_model=true."
        )

    if train_rows[0]["split"] != "train":
        raise ValueError(
            "The training condition must use split='train'."
        )

    if len(reference_rows) != 1:
        raise ValueError(
            "Exactly one condition must have is_reference=true."
        )

    if reference_rows[0]["split"] != "test":
        raise ValueError(
            "The reference condition must use split='test'."
        )


def get_training_condition(config: dict) -> dict:
    return next(
        condition
        for condition in config["conditions"]
        if condition["train_model"]
    )


def get_evaluation_conditions(config: dict) -> list[dict]:
    return [
        condition
        for condition in config["conditions"]
        if condition["evaluate"]
    ]


def get_reference_condition(config: dict) -> dict:
    return next(
        condition
        for condition in config["conditions"]
        if condition["is_reference"]
    )


def snapshot_metadata(
    data_root: Path,
    config_path: Path,
    config: dict,
) -> None:
    """Copy the active config and protocol into the run metadata folder."""
    run_paths = get_run_paths(data_root, config)
    metadata_dir = run_paths["metadata_dir"]
    ensure_dir(metadata_dir)

    protocol_path = data_root / "04_protocols" / "text_condition_protocol.csv"

    snapshot_config_path = metadata_dir / "experiment_config_snapshot.json"
    snapshot_protocol_path = metadata_dir / "text_condition_protocol_snapshot.csv"

    shutil.copy2(config_path, snapshot_config_path)

    if protocol_path.exists():
        shutil.copy2(protocol_path, snapshot_protocol_path)

    artifact_manifest = {
        "experiment_id": config["experiment_id"],
        "run_dir": config["run_dir"],
        "conditions": config["conditions"],
        "outputs": config["outputs"],
    }

    manifest_path = metadata_dir / "artifact_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(artifact_manifest, file, indent=2)
        file.write("\n")

    readme_path = run_paths["run_dir"] / "README.txt"
    if not readme_path.exists():
        train_condition = get_training_condition(config)
        evaluation_conditions = get_evaluation_conditions(config)
        evaluation_names = ", ".join(
            condition["name"]
            for condition in evaluation_conditions
        )

        readme_lines = [
            f"Experiment ID: {config['experiment_id']}",
            "",
            "Fixed inputs:",
            f"- train text input: {config['fixed_inputs']['train_text_input']}",
            f"- test text input: {config['fixed_inputs']['test_text_input']}",
            f"- pseudo-outcomes: {config['fixed_inputs']['pseudo_outcomes']}",
            f"- test causal data: {config['fixed_inputs']['test_causal_data']}",
            "",
            f"Training condition: {train_condition['name']} ({train_condition['split']})",
            f"Evaluation conditions: {evaluation_names}",
            "",
            "The active config and text protocol are copied to metadata/.",
        ]

        ensure_dir(readme_path.parent)
        readme_path.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")


def validate_patient_ids(
    data_frame: pd.DataFrame,
    label: str,
) -> None:
    if "patient_id" not in data_frame.columns:
        raise ValueError(f"{label} does not contain patient_id.")

    if data_frame["patient_id"].isna().any():
        raise ValueError(f"{label} contains missing patient_id values.")

    if data_frame["patient_id"].duplicated().any():
        raise ValueError(f"{label} contains duplicate patient_id values.")
