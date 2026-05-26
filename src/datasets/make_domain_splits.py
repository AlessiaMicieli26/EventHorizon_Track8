import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def patient_split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    subjects = df[["subject", "disease_label"]].drop_duplicates()
    stratify = subjects["disease_label"] if subjects["disease_label"].nunique() > 1 else None
    train_subjects, test_subjects = train_test_split(
        subjects,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    train_df = df[df["subject"].isin(train_subjects["subject"])].copy()
    test_df = df[df["subject"].isin(test_subjects["subject"])].copy()
    return train_df, test_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Create source/target scanner-domain splits.")
    parser.add_argument("--index-csv", default="data/processed/00_prepared/index_prepared.csv")
    parser.add_argument("--out-dir", default="data/processed/01_domain_split")
    parser.add_argument("--source-manufacturer", default="auto")
    parser.add_argument("--target-manufacturer", default="auto")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--target-adapt-size", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.index_csv)
    counts = df.groupby("manufacturer")["subject"].nunique().sort_values(ascending=False)
    if len(counts) < 2:
        raise RuntimeError("Need at least two scanner manufacturers for domain adaptation.")

    source = counts.index[0] if args.source_manufacturer == "auto" else args.source_manufacturer.upper()
    target = counts.index[1] if args.target_manufacturer == "auto" else args.target_manufacturer.upper()
    if source == target:
        raise RuntimeError("Source and target manufacturers must be different.")

    source_df = df[df["manufacturer"] == source].copy()
    target_df = df[df["manufacturer"] == target].copy()
    source_train, source_val = patient_split(source_df, args.val_size, args.seed)
    target_adapt, target_test = patient_split(target_df, 1.0 - args.target_adapt_size, args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_train.to_csv(out_dir / "source_train.csv", index=False)
    source_val.to_csv(out_dir / "source_val.csv", index=False)
    target_adapt.to_csv(out_dir / "target_adapt_unlabeled.csv", index=False)
    target_test.to_csv(out_dir / "target_test.csv", index=False)

    print(f"Source manufacturer: {source}")
    print(f"Target manufacturer: {target}")
    for name, split in [
        ("source_train", source_train),
        ("source_val", source_val),
        ("target_adapt_unlabeled", target_adapt),
        ("target_test", target_test),
    ]:
        print(f"\n{name}: scans={len(split)}, subjects={split['subject'].nunique()}")
        print(split["disease_label"].value_counts())


if __name__ == "__main__":
    main()
