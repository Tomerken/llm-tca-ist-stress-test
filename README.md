# Inference-Time Text Information Loss Stress Test on IST

This repository contains a controlled stress test for text-based conditional average treatment effect (CATE) estimation.

The experiment is inspired by the inference-time text confounding setting introduced by Ma et al. (NeurIPS 2025). It does not reproduce the full LLM-TCA method or the original experiments. Instead, it investigates a focused question about the **training-to-deployment mismatch**:

> Can a treatment-effect estimation model trained on complete, standardized clinical descriptions reliably generalize to real-world deployment scenarios where patient information is less precise, incomplete, or expressed differently?
>
> More specifically, how does the mismatch between high-quality training descriptions and less informative inference-time text affect the reliability of individualized treatment-effect estimation?

This study evaluates that concern through controlled, deterministic changes to inference-time text, rather than through real-world patient-authored descriptions.

## Experiment design

The pipeline starts from structured covariates from the International Stroke Trial (IST) dataset.

1. A fixed semi-synthetic data-generating process creates treatment assignment, potential outcomes, observed outcomes, and the known individual treatment effect `true_tau`.
2. A doubly robust pseudo-outcome is constructed from the training set.
3. One neural regressor is trained on BERT embeddings of **full** training text only.
4. The unchanged model is evaluated on the same held-out patients under three text conditions:
   - `full`: all selected text-visible covariates are represented.
   - `sparse`: four medical-history variables are omitted.
   - `general`: age and blood pressure are coarsened, and clinical information is substantially reduced.
5. Because `true_tau` is known by construction, CATE recovery is evaluated with PEHE.

The causal data, train/test split, pseudo-outcomes, and model configuration are fixed across the three conditions. Only the inference-time text changes.

## Final results

| Text condition | PEHE |
|---|---:|
| Full | 0.650 |
| Sparse | 1.547 |
| General | 3.684 |

The results show monotonic degradation in CATE recovery across the three predefined text-information conditions.

**Note:** These values are from the included single-seed experiment (**seed 42**).

## Repository structure

```text
.
├── README.md
├── requirements.txt
└── IST_based_stress_test/
    ├── 00_raw/                      # IST source files used by the pipeline
    ├── 01_semisynthetic_data/       # fixed semi-synthetic train/test datasets
    ├── 02_dr_pseudo_outcomes/       # fixed training pseudo-outcomes
    ├── 03_text_inputs/              # fixed feature-only text inputs
    ├── 04_protocols/                # text protocol and experiment configuration
    ├── scripts/                      # pipeline scripts
    └── runs/
        └── all_active_information_loss/
            ├── text/
            ├── embeddings/
            ├── results/
            └── metadata/
```

## Running the pipeline

Recommended environment: **Python 3.11**. The repository uses pinned dependencies listed in `requirements.txt`.

Create a virtual environment and install the dependencies (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Run the scripts from the repository root in this order:

```powershell
python .\IST_based_stress_test\scripts\01_construct_semisynthetic_data.py
python .\IST_based_stress_test\scripts\02_construct_pseudo_outcomes.py
python .\IST_based_stress_test\scripts\03_create_text_inputs.py
python .\IST_based_stress_test\scripts\04_generate_text_conditions.py
python .\IST_based_stress_test\scripts\05_create_embeddings.py
python .\IST_based_stress_test\scripts\06_validate_embeddings.py
python .\IST_based_stress_test\scripts\07_train_and_evaluate_text_model.py
python .\IST_based_stress_test\scripts\08_analyze_embedding_shift.py
```

Scripts 01–03 build the fixed causal-data layer. Scripts 04–08 generate the text conditions, embeddings, CATE predictions, and embedding-shift diagnostics.

Existing artifacts are protected by default. To intentionally regenerate an existing output, change the relevant script's `OVERWRITE_EXISTING` setting.

## Main outputs

The final experiment artifacts are stored in:

```text
IST_based_stress_test/runs/all_active_information_loss/
```

Key result files:

```text
results/text_condition_results.csv
results/text_condition_predictions.csv
results/embedding_shift_results.csv
```

## References and acknowledgements

This stress test is inspired by:

- Yuchen Ma, Dennis Frauen, Jonas Schweisthal, and Stefan Feuerriegel. *LLM-Driven Treatment Effect Estimation Under Inference Time Text Confounding*. NeurIPS 2025.  
  Paper: <https://proceedings.neurips.cc/paper_files/paper/2025/hash/0f3b62a67900418cbb8583839129ffc6-Abstract-Conference.html>

- Original LLM-TCA implementation: <https://github.com/yccm/llm-tca>

The experiment uses the **International Stroke Trial database (version 2)**:

- Sandercock, Peter; Niewada, Maciej; Czlonkowska, Anna. (2011). *International Stroke Trial database (version 2)* [dataset]. University of Edinburgh, Department of Clinical Neurosciences. <https://doi.org/10.7488/ds/104>
- Dataset record and source files: <https://datashare.ed.ac.uk/handle/10283/124>
- P. A. G. Sandercock et al. *The International Stroke Trial database*. *Trials*, 2011. <https://pmc.ncbi.nlm.nih.gov/articles/PMC3104487/>

**Dataset license:** The original IST dataset is made available under the [Open Data Commons Attribution License (ODC-By) v1.0](https://opendatacommons.org/licenses/by/1-0/). The original IST source files in `IST_based_stress_test/00_raw/` are included for reproducibility. The semi-synthetic datasets and derived text representations in this repository were created for this experiment using IST covariates. When reusing or redistributing the original data or derived databases, retain the required attribution and consult the original dataset's license terms.

## Scope

This repository contains one final experiment:

```text
all_active_information_loss
```
