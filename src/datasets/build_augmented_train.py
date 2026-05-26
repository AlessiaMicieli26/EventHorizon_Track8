import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine original source scans with CycleGAN-translated source slices.")
    parser.add_argument("--source-csv", default="data/processed/01_domain_split/source_train.csv")
    parser.add_argument("--translated-csv", default="data/processed/02_translated_source/translated_source_train.csv")
    parser.add_argument("--out-csv", default="data/processed/02_translated_source/source_train_augmented.csv")
    args = parser.parse_args()

    source = pd.read_csv(args.source_csv)
    translated = pd.read_csv(args.translated_csv)
    source = source.copy()
    source["image_path"] = ""
    augmented = pd.concat([source, translated], ignore_index=True)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(out_csv, index=False)
    print(f"Saved augmented training CSV: {out_csv}")
    print(f"Original scans: {len(source)}")
    print(f"Translated slices: {len(translated)}")
    print(f"Total rows: {len(augmented)}")


if __name__ == "__main__":
    main()
