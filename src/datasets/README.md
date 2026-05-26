# Datasets

ADNI preprocessing for scanner-domain adaptation.

Run order:

```bash
python src/datasets/prepare_adni.py \
  --csv data/ADNI1_Annual_2_Yr_3T_5_26_2026.csv \
  --nifti-root data/raw/nifti \
  --metadata-root data/raw/metadata \
  --out-dir data/processed/00_prepared \
  --field 3T \
  --label-mode cn_vs_ad

python src/datasets/make_domain_splits.py \
  --index-csv data/processed/00_prepared/index_prepared.csv \
  --out-dir data/processed/01_domain_split

python src/datasets/build_augmented_train.py \
  --source-csv data/processed/01_domain_split/source_train.csv \
  --translated-csv data/processed/02_translated_source/translated_source_train.csv \
  --out-csv data/processed/02_translated_source/source_train_augmented.csv
```

`prepare_adni.py` keeps only T1 MRI scans, reads scanner metadata from IDA XML files,
chooses the most common field strength when `--field auto` is used, and saves normalized
volumes plus `index_prepared.csv`.

`make_domain_splits.py` chooses the two most common manufacturers by default and creates
source/target splits for the domain-adaptation experiment.

`build_augmented_train.py` combines original source-domain examples with CycleGAN-translated
source slices for the final classifier experiment.
