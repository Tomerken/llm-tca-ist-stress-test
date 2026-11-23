import pandas as pd

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
# Text inputs are fixed. This script only creates the declared
# text conditions for the active experiment config.
# ============================================================

DATA_ROOT = find_data_root()
CONFIG, CONFIG_PATH = load_experiment_config(DATA_ROOT)
validate_condition_config(CONFIG)
RUN_PATHS = get_run_paths(DATA_ROOT, CONFIG)
PROTOCOL_PATH = DATA_ROOT / "04_protocols" / "text_condition_protocol.csv"

OVERWRITE_EXISTING = False

TRAIN_INPUT_PATH = DATA_ROOT / CONFIG["fixed_inputs"]["train_text_input"]
TEST_INPUT_PATH = DATA_ROOT / CONFIG["fixed_inputs"]["test_text_input"]


# ============================================================
# 1. Shared input validation
# ============================================================

required_columns = [
    "patient_id",
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


def validate_input_columns(input_df: pd.DataFrame) -> None:
    missing_columns = [
        column
        for column in required_columns
        if column not in input_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    validate_patient_ids(input_df, "Text input")


def sex_sentence(row: pd.Series) -> str:
    if row["SEX"] == "M":
        return "I am a man."
    if row["SEX"] == "F":
        return "I am a woman."
    raise ValueError(f"Unexpected SEX value: {row['SEX']}")


# ============================================================
# 2. Text templates
# ------------------------------------------------------------
# The wording is the same as the previous controlled protocol.
# Only the old version names were removed.
# ============================================================

rdef_sentences = {
    "RDEF1": {
        "Y": "I feel weakness in my face.",
        "N": "I do not feel weakness in my face.",
        "C": "I am not sure whether I feel weakness in my face.",
    },
    "RDEF2": {
        "Y": "I have weakness in my arm.",
        "N": "I do not have weakness in my arm.",
        "C": "I am not sure whether I have weakness in my arm.",
    },
    "RDEF3": {
        "Y": "I have weakness in my leg.",
        "N": "I do not have weakness in my leg.",
        "C": "I am not sure whether I have weakness in my leg.",
    },
    "RDEF4": {
        "Y": "I am having trouble speaking clearly.",
        "N": "I am not having trouble speaking clearly.",
        "C": "I am not sure whether I am having trouble speaking clearly.",
    },
    "RDEF5": {
        "Y": "Part of my vision is missing.",
        "N": "I do not notice a problem with my vision.",
        "C": "I am not sure whether I have a vision problem.",
    },
    "RDEF6": {
        "Y": "I have trouble judging where things are around me.",
        "N": "I do not have trouble judging where things are around me.",
        "C": "I am not sure whether I have trouble judging where things are around me.",
    },
    "RDEF7": {
        "Y": "I have trouble with balance and coordination.",
        "N": "My balance and coordination are normal.",
        "C": "I am not sure whether my balance and coordination are normal.",
    },
    "RDEF8": {
        "Y": "I notice another unusual neurological problem.",
        "N": "I do not notice another unusual neurological problem.",
        "C": "I am not sure whether I have another unusual neurological problem.",
    },
}


def create_full_text(row: pd.Series) -> str:
    sentences = [
        f"I am {row['AGE']} years old.",
        sex_sentence(row),
        f"My blood pressure reading was {row['RSBP']} mmHg.",
    ]

    if row["RATRIAL"] == "Y":
        sentences.append("I have been told that I have atrial fibrillation.")
    else:
        sentences.append("I have not been told that I have atrial fibrillation.")

    if row["RVISINF"] == "Y":
        sentences.append("A scan showed an infarct.")
    else:
        sentences.append("A scan did not show an infarct.")

    if row["RHEP24"] == "Y":
        sentences.append("I received heparin within the previous 24 hours.")
    else:
        sentences.append("I did not receive heparin within the previous 24 hours.")

    if row["RASP3"] == "Y":
        sentences.append("I took aspirin within the previous 3 days.")
    else:
        sentences.append("I did not take aspirin within the previous 3 days.")

    for variable_name, sentence_map in rdef_sentences.items():
        sentences.append(sentence_map[row[variable_name]])

    return " ".join(sentences)


def create_sparse_text(row: pd.Series) -> str:
    sentences = [
        f"I am {row['AGE']} years old.",
        sex_sentence(row),
        f"My blood pressure reading was {row['RSBP']} mmHg.",
    ]

    # Sparse is a literal text subset of Full for shared variables.
    for variable_name, sentence_map in rdef_sentences.items():
        sentences.append(sentence_map[row[variable_name]])

    return " ".join(sentences)


def get_age_decade(age: float) -> str:
    if age < 30:
        return "twenties"
    if age < 40:
        return "thirties"
    if age < 50:
        return "forties"
    if age < 60:
        return "fifties"
    if age < 70:
        return "sixties"
    if age < 80:
        return "seventies"
    if age < 90:
        return "eighties"
    return "nineties"


def get_bp_category(rsbp: float) -> str:
    if rsbp < 100:
        return "low"
    if rsbp < 140:
        return "normal"
    return "high"


def create_general_text(row: pd.Series) -> str:
    sex_word = "man" if row["SEX"] == "M" else "woman"
    age_decade = get_age_decade(row["AGE"])
    bp_category = get_bp_category(row["RSBP"])

    sentences = [
        (
            f"I am a {sex_word} in my {age_decade}. "
            f"My blood pressure seemed {bp_category}."
        )
    ]

    if row["RDEF1"] == "Y":
        sentences.append("I feel weakness in my face.")
    if row["RDEF2"] == "Y":
        sentences.append("I have weakness in my arm.")
    if row["RDEF3"] == "Y":
        sentences.append("I have weakness in my leg.")
    if row["RDEF4"] == "Y":
        sentences.append("I am having trouble speaking clearly.")
    if row["RDEF5"] == "Y":
        sentences.append("Part of my vision is missing.")
    if row["RDEF6"] == "Y" or row["RDEF7"] == "Y":
        sentences.append("I feel generally disoriented or maybe unsteady.")
    if row["RDEF8"] == "Y":
        sentences.append("Something else also feels wrong, but I cannot describe it clearly.")

    return " ".join(sentences)


TEXT_GENERATORS = {
    "full": create_full_text,
    "sparse": create_sparse_text,
    "general": create_general_text,
}


# ============================================================
# 3. Protocol and text checks
# ============================================================


def load_protocol_conditions(protocol_path):
    if not protocol_path.exists():
        raise FileNotFoundError(f"Missing text protocol: {protocol_path}")

    protocol_df = pd.read_csv(protocol_path)

    if "condition" not in protocol_df.columns:
        raise ValueError("Text protocol must contain a condition column.")

    conditions = set()
    for value in protocol_df["condition"].dropna():
        for condition_name in str(value).split("|"):
            condition_name = condition_name.strip()
            if condition_name and condition_name != "All conditions":
                conditions.add(condition_name)

    return conditions


def validate_condition_text(condition_name: str, text_df: pd.DataFrame) -> None:
    combined_text = " ".join(text_df["text"].tolist())

    if condition_name == "sparse":
        forbidden_markers = [
            "atrial fibrillation",
            "A scan",
            "heparin",
            "aspirin",
        ]
    elif condition_name == "general":
        forbidden_markers = [
            "atrial fibrillation",
            "A scan",
            "heparin",
            "aspirin",
            "mmHg",
        ]
    else:
        forbidden_markers = []

    found_markers = [
        marker
        for marker in forbidden_markers
        if marker in combined_text
    ]

    if found_markers:
        raise ValueError(
            f"{condition_name} text contains forbidden markers: "
            f"{found_markers}"
        )


# ============================================================
# 4. Generate declared text files
# ============================================================

protocol_conditions = load_protocol_conditions(PROTOCOL_PATH)
active_conditions = {
    condition["name"]
    for condition in CONFIG["conditions"]
}

missing_protocol_conditions = active_conditions - protocol_conditions
if missing_protocol_conditions:
    raise ValueError(
        "Experiment config uses conditions missing from protocol: "
        f"{sorted(missing_protocol_conditions)}"
    )

snapshot_metadata(DATA_ROOT, CONFIG_PATH, CONFIG)

for condition in CONFIG["conditions"]:
    condition_name = condition["name"]
    generator_name = condition["generator"]
    split = condition["split"]

    if generator_name not in TEXT_GENERATORS:
        raise ValueError(f"Unknown text generator: {generator_name}")

    input_path = TRAIN_INPUT_PATH if split == "train" else TEST_INPUT_PATH
    output_path = RUN_PATHS["text_dir"] / condition["text_filename"]

    if output_path.exists() and not OVERWRITE_EXISTING:
        print(
            "[skip ] exists, not overwriting "
            f"(set OVERWRITE_EXISTING=True): {output_path}"
        )
        continue

    input_df = pd.read_csv(input_path)
    validate_input_columns(input_df)

    output_df = pd.DataFrame(
        {
            "patient_id": input_df["patient_id"],
            "text": input_df.apply(TEXT_GENERATORS[generator_name], axis=1),
        }
    )

    validate_patient_ids(output_df, f"{condition_name} {split} text")

    if len(output_df) != len(input_df):
        raise ValueError("Text output row count does not match input row count.")

    if output_df["text"].isna().any() or (output_df["text"].str.strip() == "").any():
        raise ValueError("Generated text contains missing or empty values.")

    if not output_df["patient_id"].equals(input_df["patient_id"]):
        raise ValueError("Patient IDs changed or their order was altered.")

    validate_condition_text(condition_name, output_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("Condition:", condition_name)
    print("Split:", split)
    print("[write]", output_path)
    print("Rows:", len(output_df))
    print(output_df.head(3).to_string(index=False))

print("\nFinished generating text conditions.")
