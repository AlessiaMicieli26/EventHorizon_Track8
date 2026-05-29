# Training

Training scripts for ADNI scanner-domain adaptation.

Baseline source-only classifier:

```bash
python src/training/train_classifier.py \
  --train-csv data/processed/01_domain_split/source_train.csv \
  --val-csv data/processed/01_domain_split/source_val.csv \
  --test-csv data/processed/01_domain_split/target_test.csv \
  --out-dir experiments/outputs/classifier_source_only \
  --checkpoint-dir experiments/checkpoints \
  --checkpoint-name classifier_source_only_best_model.pt
```

CycleGAN scanner adaptation:

```bash
python src/training/train_cyclegan.py \
  --source-csv data/processed/01_domain_split/source_train.csv \
  --target-csv data/processed/01_domain_split/target_adapt_unlabeled.csv \
  --out-dir experiments/outputs/cyclegan_scanner \
  --checkpoint-dir experiments/checkpoints

python -m src.training.translate_source \
  --source-csv data/processed/01_domain_split/source_train.csv \
  --generator experiments/checkpoints/cyclegan_generator_source_to_target.pt \
  --out-dir data/processed/02_translated_source

python src/datasets/build_augmented_train.py \
  --source-csv data/processed/01_domain_split/source_train.csv \
  --translated-csv data/processed/02_translated_source/translated_source_train.csv \
  --out-csv data/processed/02_translated_source/source_train_augmented.csv

python src/training/train_classifier.py \
  --train-csv data/processed/02_translated_source/source_train_augmented.csv \
  --val-csv data/processed/01_domain_split/source_val.csv \
  --test-csv data/processed/01_domain_split/target_test.csv \
  --out-dir experiments/outputs/classifier_cyclegan_augmented \
  --checkpoint-dir experiments/checkpoints \
  --checkpoint-name classifier_cyclegan_augmented_best_model.pt
```

Both classifier runs evaluate on `data/processed/01_domain_split/target_test.csv`, so the
reported target metrics compare direct scanner generalization against CycleGAN augmentation.

## VAE scanner signatures + GAN pipeline

This is the leave-one-manufacturer-out setup requested in the project notes: train on two
scanner manufacturers, synthesize extra scanner-style signatures in VAE latent space, optionally
add CycleGAN translations, then test only on the third manufacturer.

```bash
python3 src/datasets/make_domain_splits.py \
  --index-csv data/processed/00_prepared/index_prepared.csv \
  --source-manufacturers "SIEMENS,GE MEDICAL SYSTEMS" \
  --target-manufacturer "PHILIPS MEDICAL SYSTEMS" \
  --out-dir data/processed/01_domain_split

python3 src/training/train_vae_signature.py \
  --train-csv data/processed/01_domain_split/source_train.csv \
  --out-dir experiments/outputs/vae_signature \
  --checkpoint-dir experiments/checkpoints \
  --checkpoint-name vae_signature.pt \
  --epochs 40 \
  --batch-size 8

python3 src/training/generate_signature_augmented.py \
  --source-csv data/processed/01_domain_split/source_train.csv \
  --vae experiments/checkpoints/vae_signature.pt \
  --out-dir data/processed/03_vae_signature_augmented \
  --include-original \
  --copies-per-target 1 \
  --alpha 1.0 \
  --noise-scale 0.25

python3 src/training/train_cyclegan.py \
  --source-csv data/processed/01_domain_split/source_train.csv \
  --target-csv data/processed/01_domain_split/target_adapt_unlabeled.csv \
  --out-dir experiments/outputs/cyclegan_scanner \
  --checkpoint-dir experiments/checkpoints \
  --epochs 30 \
  --batch-size 4

python3 -m src.training.translate_source \
  --source-csv data/processed/01_domain_split/source_train.csv \
  --generator experiments/checkpoints/cyclegan_generator_source_to_target.pt \
  --out-dir data/processed/02_translated_source

python3 src/datasets/build_augmented_train.py \
  --source-csv data/processed/03_vae_signature_augmented/source_train_vae_signature_augmented.csv \
  --translated-csv data/processed/02_translated_source/translated_source_train.csv \
  --out-csv data/processed/04_signature_gan_augmented/source_train_signature_gan_augmented.csv

python3 src/training/train_classifier.py \
  --train-csv data/processed/04_signature_gan_augmented/source_train_signature_gan_augmented.csv \
  --val-csv data/processed/01_domain_split/source_val.csv \
  --test-csv data/processed/01_domain_split/target_test.csv \
  --out-dir experiments/outputs/classifier_signature_gan_augmented \
  --checkpoint-dir experiments/checkpoints \
  --checkpoint-name classifier_signature_gan_augmented_best_model.pt \
  --epochs 20 \
  --batch-size 8
```

For a source-only control using the same two manufacturers and the same third-manufacturer test:

```bash
python3 src/training/train_classifier.py \
  --train-csv data/processed/01_domain_split/source_train.csv \
  --val-csv data/processed/01_domain_split/source_val.csv \
  --test-csv data/processed/01_domain_split/target_test.csv \
  --out-dir experiments/outputs/classifier_two_source_only \
  --checkpoint-dir experiments/checkpoints \
  --checkpoint-name classifier_two_source_only_best_model.pt
```

## AdaIN scanner signatures comparison

AdaIN keeps the same leave-one-manufacturer-out split, but replaces VAE latent-vector image
generation with feature-statistics style transfer. The comparison is:

- VAE signatures + GAN: `classifier_signature_gan_augmented`
- AdaIN signatures + GAN: `classifier_adain_gan_augmented`

```bash
python3 src/training/train_adain_signature.py \
  --train-csv data/processed/01_domain_split/source_train.csv \
  --out-dir experiments/outputs/adain_signature \
  --checkpoint-dir experiments/checkpoints \
  --checkpoint-name adain_signature.pt \
  --epochs 40 \
  --batch-size 8

python3 src/training/generate_adain_augmented.py \
  --source-csv data/processed/01_domain_split/source_train.csv \
  --adain experiments/checkpoints/adain_signature.pt \
  --out-dir data/processed/05_adain_signature_augmented \
  --include-original \
  --alpha 1.0

python3 src/datasets/build_augmented_train.py \
  --source-csv data/processed/05_adain_signature_augmented/source_train_adain_signature_augmented.csv \
  --translated-csv data/processed/02_translated_source/translated_source_train.csv \
  --out-csv data/processed/06_adain_gan_augmented/source_train_adain_gan_augmented.csv

python3 src/training/train_classifier.py \
  --train-csv data/processed/06_adain_gan_augmented/source_train_adain_gan_augmented.csv \
  --val-csv data/processed/01_domain_split/source_val.csv \
  --test-csv data/processed/01_domain_split/target_test.csv \
  --out-dir experiments/outputs/classifier_adain_gan_augmented \
  --checkpoint-dir experiments/checkpoints \
  --checkpoint-name classifier_adain_gan_augmented_best_model.pt \
  --epochs 20 \
  --batch-size 8
```

## Cluster GPU run

Use this flow on the DMI cluster. The script stops immediately if CUDA is not visible inside
Apptainer.

```bash
ssh mcllss02d66h163m@gcluster.dmi.unict.it
cd ~/DomainAdaptation-Track9-DataLost

git remote -v
git branch --show-current
git fetch origin
git checkout Init_DL_PJ || git checkout -b Init_DL_PJ origin/Init_DL_PJ
git pull origin Init_DL_PJ

srun --account=dl-course-q2 \
  --partition=dl-course-q2 \
  --qos=gpu-medium \
  --gres=gpu:1 \
  --pty bash

cd ~/DomainAdaptation-Track9-DataLost
apptainer shell --nv /shared/sifs/latest.sif
```

Inside Apptainer:

```bash
python --version
nvidia-smi
python -c "import torch, numpy, pandas, sklearn, nibabel, matplotlib; print('deps ok')"

# Only if needed:
pip install --user nibabel matplotlib

RUN_PREPROCESS=1 bash src/training/run_cluster_vae_adain_pipeline.sh
```

Useful overrides:

```bash
# Skip preprocessing only when the prepared index already includes the desired CSV inputs.
RUN_PREPROCESS=auto bash src/training/run_cluster_vae_adain_pipeline.sh

# Reuse an existing CycleGAN generator checkpoint.
RUN_CYCLEGAN=0 bash src/training/run_cluster_vae_adain_pipeline.sh

# Short smoke test on GPU.
CLASSIFIER_EPOCHS=1 GEN_EPOCHS=1 CYCLEGAN_EPOCHS=1 bash src/training/run_cluster_vae_adain_pipeline.sh
```

Final comparison outputs:

```bash
experiments/outputs/evaluation_multi/summary.csv
experiments/outputs/evaluation_multi/summary.md
figures/multi_model_accuracy_f1.png
```

Copy results back from the cluster:

```bash
rsync -av --progress \
  mcllss02d66h163m@gcluster.dmi.unict.it:~/DomainAdaptation-Track9-DataLost/experiments/outputs/ \
  ./experiments/outputs/

rsync -av --progress \
  mcllss02d66h163m@gcluster.dmi.unict.it:~/DomainAdaptation-Track9-DataLost/experiments/checkpoints/ \
  ./experiments/checkpoints/

rsync -av --progress \
  mcllss02d66h163m@gcluster.dmi.unict.it:~/DomainAdaptation-Track9-DataLost/figures/ \
  ./figures/
```
