---
license: cc-by-4.0
pretty_name: Nonastreda Multimodal Tool Wear Dataset
size_categories:
- n<1K
task_categories:
- image-classification
- tabular-regression
tags:
- multimodal
- multi-image
- multi-output-regression
- multilabel-regression
- classification
- regression
- tool-wear
- condition-monitoring
- predictive-maintenance
- manufacturing
- milling
- machining
- industrial-ai
- image
- tabular
- timeseries
- force-signals
- cc-by-4.0
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
---

# Nonastreda: Multimodal Dataset for Tool Wear State Monitoring

**Nonastreda** is a multimodal dataset for efficient tool wear state monitoring in milling. It contains 512 sample-level records combining visual, time-frequency, and force-signal-derived representations of tool wear.

This Hugging Face repository is a mirror and machine-learning-friendly access point for the dataset. The canonical scholarly description is the associated *Data in Brief* article, and the canonical archived dataset record is hosted on Mendeley Data.

## Official links

- **Data descriptor article:** [Nonastreda multimodal dataset for efficient tool wear state monitoring](https://www.sciencedirect.com/science/article/pii/S2352340925006298)
- **Archived dataset:** [Mendeley Data record](https://data.mendeley.com/datasets/m892d2wtzh/1)
- **Mendeley Data DOI:** `10.17632/m892d2wtzh.1`
- **License:** Creative Commons Attribution 4.0 International, CC BY 4.0

## Overview

![Nonastreda overview pipeline](figures/overview_pipeline.png)

The dataset combines visual inspection data, force-signal-derived image representations, and raw force signals for tool wear monitoring. Each sample is indexed by an identifier such as `T10R10B1`, and the same identifier is used across the available image modalities.

Each sample-level row includes:

- `tool_id` and `run_id`, parsed from the sample identifier,
- one microscope-image-based tool-wear classification label,
- one additional force-phase classification label derived from force-signal amplitudes,
- three regression targets,
- nine image modalities.

The raw force-signal file is also included for future research, but it is not used as a sample-level visualization table because the sequences are organized per tool rather than cut into sub-sequences for each blade/image sample.

## Modalities

Each sample is represented by nine image modalities:

| Modality group | Viewer columns | Description |
|---|---|---|
| Visual images | `chip`, `tool`, `work` | Images of chip, tool blade, and workpiece |
| Mel-spectrograms | `spec_x`, `spec_y`, `spec_z` | Mel-spectrogram representations of force-signal axes X, Y, and Z |
| Scalograms | `scal_x`, `scal_y`, `scal_z` | Scalogram representations of force-signal axes X, Y, and Z |

The human-readable `metadata.csv` file keeps the corresponding file path columns, such as `chip_file_name`, `tool_file_name`, and `scal_z_file_name`.

## Dataset split and recommended evaluation protocol

This repository provides a single Hugging Face split:

- `train`: all 512 samples.

No fixed train/test split is imposed. For model evaluation, we recommend **leave-one-tool-out cross-validation** using `tool_id`.

Since the dataset contains 10 tools, this gives 10 train/test folds. In each fold, samples from one tool are held out for testing, and samples from the remaining nine tools are used for training. This protocol evaluates whether a model generalizes to unseen tools rather than only learning tool-specific visual or signal patterns.

Example fold definition:

```python
from datasets import load_dataset

NUM_TOOLS = 10

ds = load_dataset("hubtru/nonastreda", split="train")

for held_out_tool in range(1, NUM_TOOLS + 1):
    train_ds = ds.filter(lambda x: x["tool_id"] != held_out_tool)
    test_ds = ds.filter(lambda x: x["tool_id"] == held_out_tool)
    print(f"Fold T{held_out_tool}: train={len(train_ds)}, test={len(test_ds)}")
```

## Tasks

This dataset can be used for several related machine-learning tasks.

### 1. Image-based tool wear classification

Recommended primary classification task:

- input: one or more of the nine image modalities,
- target: `image_label` or `image_label_id`,
- classes: `sharp`, `used`, `dulled`.

### 2. Multimodal multi-output regression

Recommended regression task:

- input: one or more of the nine image modalities,
- targets: `gaps`, `flank_wear`, and `overhang`.

### 3. Exploratory force-phase classification

Additional research task:

- input: force-related representations or raw force signals,
- target: `force_phase_label` or `force_phase_name`,
- phases: initial/break-in wear, steady-state/stabilization wear, accelerated/severe wear.

This label is provided for exploratory and multimodal research and should not be treated as a direct duplicate of the image-based wear label.

## Classification labels

The dataset provides two classification label sets that represent different interpretations of tool-wear state.

### Image-based wear label

![Microscope-based tool inspection](figures/microscope_setup.png)

The `image_label` column is the primary classification label described in the associated *Data in Brief* article. It was assigned from microscope images of the tool blade using three tool-wear ranges:

- `sharp`
- `used`
- `dulled`

For convenience, the dataset also provides `image_label_id`:

- `sharp = 1`
- `used = 2`
- `dulled = 3`

This is the recommended label for reproducing the classification task described in the article.

### Force-phase label

![Force components for Tool T1](figures/force_components_t1.png)

The `force_phase_label` column is an additional research label derived from force-signal amplitudes in `forces_xyz_raw.mat`. It follows three phases of tool wear:

- `1`: initial / break-in wear
- `2`: steady-state / stabilization wear
- `3`: accelerated / severe wear

The corresponding text label is provided in `force_phase_name`.

This label is provided for exploratory and multimodal research. It was not used as the primary classification label in the *Data in Brief* article.

### Label synchronization

The column `labels_in_sync` indicates whether the image-based label and force-phase label agree numerically:

```text
labels_in_sync = image_label_id == force_phase_label
```

There are 71 samples where the two label sets are out of sync. These cases should not be interpreted as annotation errors. They reflect the fact that the labels are derived from different sources and capture different aspects of the tool-wear process.

Researchers may use these out-of-sync samples to study disagreement between visual tool-wear state and force-signal-based wear phase.

## Regression targets

The dataset provides three regression targets:

| Column | Description |
|---|---|
| `gaps` | Gap-related measurement |
| `flank_wear` | Flank wear measurement |
| `overhang` | Overhang measurement |

These targets are provided in `metadata.csv` and originate from `labels_reg.csv`.

## Repository structure

```text
nonastreda/
├── README.md
├── CITATION.cff
├── LICENSE
├── metadata.csv
├── metadata.parquet
├── data/
│   └── train-00000-of-00001.parquet
├── chip/
├── tool/
├── work/
├── spec/
│   ├── x/
│   ├── y/
│   └── z/
├── scal/
│   ├── x/
│   ├── y/
│   └── z/
├── figures/
│   ├── overview_pipeline.png
│   ├── microscope_setup.png
│   └── force_components_t1.png
└── raw/
    ├── forces_xyz_raw.mat
    ├── labels.csv
    └── labels_reg.csv
```

## Metadata columns

`metadata.csv` is the recommended human-readable index. The default Hugging Face Dataset Viewer is configured to use `data/train-*.parquet`, where the nine image modalities are stored as image columns rather than file-name strings.

| Column | Description |
|---|---|
| `sample_id` | Unique sample identifier |
| `tool_id` | Tool identifier parsed from `sample_id`, e.g. `T10R14B3` → `10` |
| `run_id` | Run identifier parsed from `sample_id`, e.g. `T10R14B3` → `14` |
| `image_label` | Primary microscope-image-based class label: `sharp`, `used`, or `dulled` |
| `image_label_id` | Numeric encoding of `image_label`: 1, 2, or 3 |
| `force_phase_label` | Additional force-amplitude-based wear phase label: 1, 2, or 3 |
| `force_phase_name` | Text description of the force-phase label |
| `labels_in_sync` | Whether `image_label_id` and `force_phase_label` agree |
| `gaps` | Regression target |
| `flank_wear` | Regression target |
| `overhang` | Regression target |
| `chip_file_name` | Path to chip image in `metadata.csv` |
| `tool_file_name` | Path to tool blade image in `metadata.csv` |
| `work_file_name` | Path to workpiece image in `metadata.csv` |
| `spec_x_file_name` | Path to X-axis Mel-spectrogram image in `metadata.csv` |
| `spec_y_file_name` | Path to Y-axis Mel-spectrogram image in `metadata.csv` |
| `spec_z_file_name` | Path to Z-axis Mel-spectrogram image in `metadata.csv` |
| `scal_x_file_name` | Path to X-axis scalogram image in `metadata.csv` |
| `scal_y_file_name` | Path to Y-axis scalogram image in `metadata.csv` |
| `scal_z_file_name` | Path to Z-axis scalogram image in `metadata.csv` |

## Raw files

The original source files are preserved under `raw/`:

- `raw/labels.csv`: original classification labels, including `image_label` and `tool_label`.
- `raw/labels_reg.csv`: original regression labels.
- `raw/forces_xyz_raw.mat`: raw force signals for the X, Y, and Z axes.

The raw `.mat` file is included to support future research on signal processing and sequence modeling. It is not used directly in the default sample-level Hugging Face Dataset Viewer configuration.

## Loading the dataset

```python
from datasets import load_dataset

ds = load_dataset("hubtru/nonastreda")
print(ds["train"][0])
```

To load the single split directly:

```python
from datasets import load_dataset

ds = load_dataset("hubtru/nonastreda", split="train")
print(ds[0])
```

Example image access:

```python
from datasets import load_dataset

ds = load_dataset("hubtru/nonastreda", split="train")
row = ds[0]

chip_image = row["chip"]
tool_image = row["tool"]
work_image = row["work"]
```

To load the human-readable metadata table after cloning or downloading the repository:

```python
import pandas as pd

metadata = pd.read_csv("metadata.csv")
print(metadata.head())
```

## Intended use

This dataset is intended for research in:

- tool wear monitoring,
- condition monitoring,
- predictive maintenance,
- industrial machine learning,
- multimodal learning,
- visual inspection,
- time-frequency image analysis,
- manufacturing process monitoring.

## Limitations and notes

- The dataset contains 512 sample-level records.
- The Hugging Face `train` split contains all samples; no fixed test split is imposed.
- For evaluation, leave-one-tool-out cross-validation using `tool_id` is recommended.
- The default Hugging Face viewer is centered on sample-level image modalities and labels.
- The raw force signals are included for completeness but require separate handling.
- The two classification labels are derived from different sources and should not be assumed to be equivalent.
- The `image_label` / `image_label_id` fields should be used for reproducing the primary classification task described in the article.
- The `force_phase_label` field should be treated as an additional exploratory label.

## License

This dataset is released under the Creative Commons Attribution 4.0 International license, CC BY 4.0.

Users may share and adapt the dataset, including for commercial purposes, provided that appropriate credit is given.

## Citation

If you use this dataset, please cite the associated *Data in Brief* article and the archived Mendeley Data record.

### Article

```bibtex
@article{truchan2025nonastreda,
  title = {Nonastreda multimodal dataset for efficient tool wear state monitoring},
  journal = {Data in Brief},
  year = {2025},
  doi = {10.1016/j.dib.2025.111905},
  url = {https://www.sciencedirect.com/science/article/pii/S2352340925006298}
}
```

### Dataset

```bibtex
@dataset{truchan2025nonastreda_mendeley,
  title = {Nonastreda: Multimodal Dataset for Identifying Tool Wear Condition},
  year = {2025},
  publisher = {Mendeley Data},
  doi = {10.17632/m892d2wtzh.1},
  url = {https://data.mendeley.com/datasets/m892d2wtzh/1}
}
```

## Contact

For questions about the dataset, please refer to the contact information provided in the associated *Data in Brief* article and Mendeley Data record.
