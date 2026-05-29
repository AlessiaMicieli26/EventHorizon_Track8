# ADNI Scanner-Domain Adaptation with CycleGAN

[![Report](https://img.shields.io/badge/Paper-REPORT.md-blue)](docs/REPORT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Group and Project Information

- **Group ID**: TBD
- **Project ID**: TBD

## Project Description

This project studies scanner-domain shift in ADNI structural MRI data. The pipeline prepares T1 MRI volumes, defines source and target domains from scanner manufacturers, trains a source-only disease classifier, and then uses a lightweight CycleGAN to translate source-domain slices toward the target scanner style. The final comparison is between direct target-domain generalization and CycleGAN-augmented training.

For the theoretical discussion, experimental analysis, and contribution declaration, use [docs/REPORT.md](docs/REPORT.md).

## Repository Layout

- `data/`: local ADNI archives, extracted raw files, and processed CSV/NumPy artifacts.
- `src/datasets/`: ADNI preparation, patient-level domain split, and augmented CSV construction.
- `src/models/`: PyTorch model definitions used by training and translation.
- `src/training/`: classifier training, CycleGAN training, and source-to-target translation.
- `src/evaluation/`: metric aggregation and experiment comparison scripts.
- `experiments/configs/`: reproducibility configuration files.
- `experiments/checkpoints/`: trained model checkpoints.
- `experiments/outputs/`: run outputs, metrics, and evaluation summaries.
- `docs/`: final report and presentation material.

## Environment Setup

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate dl-project
```

If running with the system interpreter instead of Conda, use `python3` and make sure the dependencies in `environment.yml` are installed. `nibabel` is required for NIfTI loading.

## Data Setup

Expected local inputs:

- MRI archive: `data/ADNI1_Annual 2 Yr 3T.zip`
- IDA metadata archive: `data/ADNI1_Annual_2_Yr_3T_IDA_Metadata.zip`
- Additional MRI archive: `data/ADNI1_Complete 3Yr 3T.zip`
- Additional IDA metadata archive: `data/ADNI1_Complete_3Yr_3T_IDA_Metadata.zip`
- ADNI image collection CSVs containing at least `Image Data ID`, `Subject`, `Group`, `Description`, and `Acq Date`

Extract the raw data:

```bash
mkdir -p data/raw/nifti data/raw/metadata
unzip -n "data/ADNI1_Annual 2 Yr 3T.zip" -d data/raw/nifti
unzip -n "data/ADNI1_Annual_2_Yr_3T_IDA_Metadata.zip" -d data/raw/metadata
unzip -n "data/ADNI1_Complete 3Yr 3T.zip" -d data/raw/nifti
unzip -n "data/ADNI1_Complete_3Yr_3T_IDA_Metadata.zip" -d data/raw/metadata
```

## Pipeline

### 1. Prepare ADNI Volumes

This step filters T1 scans, joins scanner metadata, normalizes volumes, and writes `.npy` volumes plus `index_prepared.csv`.

```bash
python src/datasets/prepare_adni.py \
  --csv data/ADNI1_Annual_2_Yr_3T_5_26_2026.csv \
  --csv data/ADNI1_Complete_3Yr_3T_5_29_2026.csv \
  --nifti-root data/raw/nifti \
  --metadata-root data/raw/metadata \
  --out-dir data/processed/00_prepared \
  --field 3T \
  --label-mode cn_vs_ad
```

Use `--dry-run` first to verify counts without writing all volumes:

```bash
python src/datasets/prepare_adni.py \
  --csv data/ADNI1_Annual_2_Yr_3T_5_26_2026.csv \
  --csv data/ADNI1_Complete_3Yr_3T_5_29_2026.csv \
  --nifti-root data/raw/nifti \
  --metadata-root data/raw/metadata \
  --field 3T \
  --label-mode cn_vs_ad \
  --dry-run
```

### 2. Create Patient-Level Source/Target Splits

The split uses scanner manufacturer as the domain. The most frequent manufacturer becomes the source by default, and the second most frequent becomes the target.

```bash
python src/datasets/make_domain_splits.py \
  --index-csv data/processed/00_prepared/index_prepared.csv \
  --out-dir data/processed/01_domain_split
```

Generated files:

- `source_train.csv`
- `source_val.csv`
- `target_adapt_unlabeled.csv`
- `target_test.csv`

### 3. Train Source-Only Baseline

```bash
python src/training/train_classifier.py \
  --train-csv data/processed/01_domain_split/source_train.csv \
  --val-csv data/processed/01_domain_split/source_val.csv \
  --test-csv data/processed/01_domain_split/target_test.csv \
  --out-dir experiments/outputs/classifier_source_only \
  --checkpoint-dir experiments/checkpoints \
  --checkpoint-name classifier_source_only_best_model.pt \
  --epochs 20 \
  --batch-size 8
```

The output metric file is:

```text
experiments/outputs/classifier_source_only/metrics.json
```

### 4. Train CycleGAN Scanner Adapter

```bash
python src/training/train_cyclegan.py \
  --source-csv data/processed/01_domain_split/source_train.csv \
  --target-csv data/processed/01_domain_split/target_adapt_unlabeled.csv \
  --out-dir experiments/outputs/cyclegan_scanner \
  --checkpoint-dir experiments/checkpoints \
  --epochs 30 \
  --batch-size 4
```

This trains unpaired source-to-target and target-to-source generators over center MRI slices.

### 5. Translate Source Slices

```bash
python -m src.training.translate_source \
  --source-csv data/processed/01_domain_split/source_train.csv \
  --generator experiments/outputs/cyclegan_scanner/generator_source_to_target.pt \
  --out-dir data/processed/02_translated_source
```

### 6. Build Augmented Training CSV

```bash
python src/datasets/build_augmented_train.py \
  --source-csv data/processed/01_domain_split/source_train.csv \
  --translated-csv data/processed/02_translated_source/translated_source_train.csv \
  --out-csv data/processed/02_translated_source/source_train_augmented.csv
```

### 7. Train CycleGAN-Augmented Classifier

```bash
python src/training/train_classifier.py \
  --train-csv data/processed/02_translated_source/source_train_augmented.csv \
  --val-csv data/processed/01_domain_split/source_val.csv \
  --test-csv data/processed/01_domain_split/target_test.csv \
  --out-dir experiments/outputs/classifier_cyclegan_augmented \
  --checkpoint-dir experiments/checkpoints \
  --checkpoint-name classifier_cyclegan_augmented_best_model.pt \
  --epochs 20 \
  --batch-size 8
```

### 8. Evaluate Experiments

```bash
python src/evaluation/evaluate_pipeline.py \
  --baseline experiments/outputs/classifier_source_only/metrics.json \
  --augmented experiments/outputs/classifier_cyclegan_augmented/metrics.json \
  --out-dir experiments/outputs/evaluation
```

Compare these metrics files:

- `experiments/outputs/classifier_source_only/metrics.json`
- `experiments/outputs/classifier_cyclegan_augmented/metrics.json`

Evaluation summaries are saved to:

- `experiments/outputs/evaluation/summary.csv`
- `experiments/outputs/evaluation/summary.json`
- `experiments/outputs/evaluation/summary.md`

Evaluation figures are saved to:

- `figures/metric_scatter_accuracy_f1.png`
- `figures/classifier_temporal_metrics.png`
- `figures/cyclegan_temporal_losses.png`
- `figures/qualitative_translation_previews.png`

## Current Execution Status

The pipeline has been executed locally:

- `data/raw/nifti`: 306 extracted `.nii` files
- `data/raw/metadata`: 612 extracted `.xml` files
- `data/processed/00_prepared/index_prepared.csv`: 173 prepared CN/AD T1 volumes
- source manufacturer: `SIEMENS`
- target manufacturer: `PHILIPS MEDICAL SYSTEMS`
- source-only target accuracy: `0.7429`, macro-F1: `0.7395`
- CycleGAN-augmented target accuracy: `0.5714`, macro-F1: `0.5585`
- final comparison: source-only performed better in this run
