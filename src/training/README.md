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
