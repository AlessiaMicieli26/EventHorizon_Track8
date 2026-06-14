#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PREPROCESS="${RUN_PREPROCESS:-1}"
RUN_CYCLEGAN="${RUN_CYCLEGAN:-1}"
SOURCE_MANUFACTURERS="${SOURCE_MANUFACTURERS:-SIEMENS,GE MEDICAL SYSTEMS}"
TARGET_MANUFACTURER="${TARGET_MANUFACTURER:-PHILIPS MEDICAL SYSTEMS}"
CLASSIFIER_EPOCHS="${CLASSIFIER_EPOCHS:-20}"
CLASSIFIER_BATCH_SIZE="${CLASSIFIER_BATCH_SIZE:-8}"
GEN_EPOCHS="${GEN_EPOCHS:-40}"
VAE_EPOCHS="${VAE_EPOCHS:-$GEN_EPOCHS}"
ADAIN_EPOCHS="${ADAIN_EPOCHS:-$GEN_EPOCHS}"
CYCLEGAN_EPOCHS="${CYCLEGAN_EPOCHS:-30}"
CYCLEGAN_BATCH_SIZE="${CYCLEGAN_BATCH_SIZE:-4}"

mkdir -p experiments/logs experiments/checkpoints figures

extract_archive() {
  local archive="$1"
  local destination="$2"
  if [[ -f "${archive}" ]]; then
    mkdir -p "${destination}"
    unzip -n "${archive}" -d "${destination}"
  else
    echo "Missing optional archive: ${archive}"
  fi
}

run_step() {
  local name="$1"
  shift
  echo "========== ${name} =========="
  "$@" 2>&1 | tee "experiments/logs/${name}.log"
}

echo "Repository: $(pwd)"
echo "Python: $(${PYTHON_BIN} --version)"
nvidia-smi
${PYTHON_BIN} - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible inside this container/session.")
print("device_name", torch.cuda.get_device_name(0))
PY

${PYTHON_BIN} - <<'PY'
import torch, numpy, pandas, sklearn, nibabel, matplotlib
print("deps ok")
PY

run_step extract_adni_archives extract_archive "data/ADNI1_Annual 2 Yr 3T.zip" data/raw/nifti
run_step extract_adni_annual_metadata extract_archive "data/ADNI1_Annual_2_Yr_3T_IDA_Metadata.zip" data/raw/metadata
run_step extract_adni_complete_3yr extract_archive "data/ADNI1_Complete 3Yr 3T.zip" data/raw/nifti
run_step extract_adni_complete_3yr_metadata extract_archive "data/ADNI1_Complete_3Yr_3T_IDA_Metadata.zip" data/raw/metadata

if [[ "${RUN_PREPROCESS}" == "1" || ! -f data/processed/00_prepared/index_prepared.csv ]]; then
  run_step prepare_adni \
    "${PYTHON_BIN}" src/datasets/prepare_adni.py \
      --csv data/ADNI1_Annual_2_Yr_3T_5_26_2026.csv \
      --csv data/ADNI1_Complete_3Yr_3T_5_29_2026.csv \
      --nifti-root data/raw/nifti \
      --metadata-root data/raw/metadata \
      --out-dir data/processed/00_prepared \
      --field 3T \
      --label-mode cn_vs_ad
fi

run_step make_domain_splits \
  "${PYTHON_BIN}" src/datasets/make_domain_splits.py \
    --index-csv data/processed/00_prepared/index_prepared.csv \
    --source-manufacturers "${SOURCE_MANUFACTURERS}" \
    --target-manufacturer "${TARGET_MANUFACTURER}" \
    --out-dir data/processed/01_domain_split

run_step classifier_source_only \
  "${PYTHON_BIN}" src/training/train_classifier.py \
    --train-csv data/processed/01_domain_split/source_train.csv \
    --val-csv data/processed/01_domain_split/source_val.csv \
    --test-csv data/processed/01_domain_split/target_test.csv \
    --out-dir experiments/outputs/classifier_source_only \
    --checkpoint-dir experiments/checkpoints \
    --checkpoint-name classifier_source_only_best_model.pt \
    --epochs "${CLASSIFIER_EPOCHS}" \
    --batch-size "${CLASSIFIER_BATCH_SIZE}"

run_step classifier_target_oracle \
  "${PYTHON_BIN}" src/training/train_classifier.py \
    --train-csv data/processed/01_domain_split/target_test.csv \
    --val-csv data/processed/01_domain_split/target_test.csv \
    --test-csv data/processed/01_domain_split/target_test.csv \
    --out-dir experiments/outputs/classifier_target_oracle \
    --checkpoint-dir experiments/checkpoints \
    --checkpoint-name classifier_target_oracle_best_model.pt \
    --epochs "${CLASSIFIER_EPOCHS}" \
    --batch-size "${CLASSIFIER_BATCH_SIZE}"

if [[ "${RUN_CYCLEGAN}" == "1" || ! -f experiments/checkpoints/cyclegan_generator_source_to_target.pt ]]; then
  run_step train_cyclegan \
    "${PYTHON_BIN}" src/training/train_cyclegan.py \
      --source-csv data/processed/01_domain_split/source_train.csv \
      --target-csv data/processed/01_domain_split/target_adapt_unlabeled.csv \
      --out-dir experiments/outputs/cyclegan_scanner \
      --checkpoint-dir experiments/checkpoints \
      --epochs "${CYCLEGAN_EPOCHS}" \
      --batch-size "${CYCLEGAN_BATCH_SIZE}"
fi

run_step translate_source \
  "${PYTHON_BIN}" -m src.training.translate_source \
    --source-csv data/processed/01_domain_split/source_train.csv \
    --generator experiments/checkpoints/cyclegan_generator_source_to_target.pt \
    --out-dir data/processed/02_translated_source

run_step build_cyclegan_augmented \
  "${PYTHON_BIN}" src/datasets/build_augmented_train.py \
    --source-csv data/processed/01_domain_split/source_train.csv \
    --translated-csv data/processed/02_translated_source/translated_source_train.csv \
    --out-csv data/processed/02_translated_source/source_train_augmented.csv

run_step classifier_cyclegan_augmented \
  "${PYTHON_BIN}" src/training/train_classifier.py \
    --train-csv data/processed/02_translated_source/source_train_augmented.csv \
    --val-csv data/processed/01_domain_split/source_val.csv \
    --test-csv data/processed/01_domain_split/target_test.csv \
    --out-dir experiments/outputs/classifier_cyclegan_augmented \
    --checkpoint-dir experiments/checkpoints \
    --checkpoint-name classifier_cyclegan_augmented_best_model.pt \
    --epochs "${CLASSIFIER_EPOCHS}" \
    --batch-size "${CLASSIFIER_BATCH_SIZE}"

run_step train_vae_signature \
  "${PYTHON_BIN}" src/training/train_vae_signature.py \
    --train-csv data/processed/01_domain_split/source_train.csv \
    --out-dir experiments/outputs/vae_signature \
    --checkpoint-dir experiments/checkpoints \
    --checkpoint-name vae_signature.pt \
    --epochs "${VAE_EPOCHS}" \
    --batch-size "${CLASSIFIER_BATCH_SIZE}"

run_step generate_vae_signature \
  "${PYTHON_BIN}" src/training/generate_signature_augmented.py \
    --source-csv data/processed/01_domain_split/source_train.csv \
    --vae experiments/checkpoints/vae_signature.pt \
    --out-dir data/processed/03_vae_signature_augmented \
    --include-original \
    --copies-per-target 1 \
    --alpha 1.0 \
    --noise-scale 0.10 \
    --generation-mode residual \
    --residual-scale 0.6

run_step build_vae_gan_augmented \
  "${PYTHON_BIN}" src/datasets/build_augmented_train.py \
    --source-csv data/processed/03_vae_signature_augmented/source_train_vae_signature_augmented.csv \
    --translated-csv data/processed/02_translated_source/translated_source_train.csv \
    --out-csv data/processed/04_signature_gan_augmented/source_train_signature_gan_augmented.csv

run_step classifier_vae_gan_augmented \
  "${PYTHON_BIN}" src/training/train_classifier.py \
    --train-csv data/processed/04_signature_gan_augmented/source_train_signature_gan_augmented.csv \
    --val-csv data/processed/01_domain_split/source_val.csv \
    --test-csv data/processed/01_domain_split/target_test.csv \
    --out-dir experiments/outputs/classifier_signature_gan_augmented \
    --checkpoint-dir experiments/checkpoints \
    --checkpoint-name classifier_signature_gan_augmented_best_model.pt \
    --epochs "${CLASSIFIER_EPOCHS}" \
    --batch-size "${CLASSIFIER_BATCH_SIZE}"

run_step train_adain_signature \
  "${PYTHON_BIN}" src/training/train_adain_signature.py \
    --train-csv data/processed/01_domain_split/source_train.csv \
    --out-dir experiments/outputs/adain_signature \
    --checkpoint-dir experiments/checkpoints \
    --checkpoint-name adain_signature.pt \
    --epochs "${ADAIN_EPOCHS}" \
    --batch-size "${CLASSIFIER_BATCH_SIZE}"

run_step generate_adain_signature \
  "${PYTHON_BIN}" src/training/generate_adain_augmented.py \
    --source-csv data/processed/01_domain_split/source_train.csv \
    --adain experiments/checkpoints/adain_signature.pt \
    --out-dir data/processed/05_adain_signature_augmented \
    --include-original \
    --alpha 1.0

run_step build_adain_gan_augmented \
  "${PYTHON_BIN}" src/datasets/build_augmented_train.py \
    --source-csv data/processed/05_adain_signature_augmented/source_train_adain_signature_augmented.csv \
    --translated-csv data/processed/02_translated_source/translated_source_train.csv \
    --out-csv data/processed/06_adain_gan_augmented/source_train_adain_gan_augmented.csv

run_step classifier_adain_gan_augmented \
  "${PYTHON_BIN}" src/training/train_classifier.py \
    --train-csv data/processed/06_adain_gan_augmented/source_train_adain_gan_augmented.csv \
    --val-csv data/processed/01_domain_split/source_val.csv \
    --test-csv data/processed/01_domain_split/target_test.csv \
    --out-dir experiments/outputs/classifier_adain_gan_augmented \
    --checkpoint-dir experiments/checkpoints \
    --checkpoint-name classifier_adain_gan_augmented_best_model.pt \
    --epochs "${CLASSIFIER_EPOCHS}" \
    --batch-size "${CLASSIFIER_BATCH_SIZE}"

run_step evaluate_multi \
  "${PYTHON_BIN}" src/evaluation/evaluate_multi_pipeline.py \
    --model source_only=experiments/outputs/classifier_source_only/metrics.json \
    --model target_oracle=experiments/outputs/classifier_target_oracle/metrics.json \
    --model cyclegan_augmented=experiments/outputs/classifier_cyclegan_augmented/metrics.json \
    --model vae_gan_augmented=experiments/outputs/classifier_signature_gan_augmented/metrics.json \
    --model adain_gan_augmented=experiments/outputs/classifier_adain_gan_augmented/metrics.json \
    --out-dir experiments/outputs/evaluation_multi \
    --figures-dir figures

echo "Done. Key outputs:"
ls -lh experiments/checkpoints
ls -lh experiments/outputs/evaluation_multi
ls -lh figures
