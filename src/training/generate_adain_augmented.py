import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.models.adain import ScannerAdaINAutoencoder, adaptive_instance_normalization


def load_center_slice(path: str, image_size: int) -> torch.Tensor:
    volume = np.load(path, mmap_mode="r")
    image = np.asarray(volume[:, :, volume.shape[2] // 2], dtype=np.float32)
    x = torch.from_numpy(image[None, None, ...].copy())
    if x.shape[-2:] != (image_size, image_size):
        x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return x


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create synthetic scanner-style slices by applying AdaIN scanner style signatures."
    )
    parser.add_argument("--source-csv", default="data/processed/01_domain_split/source_train.csv")
    parser.add_argument("--adain", default="experiments/checkpoints/adain_signature.pt")
    parser.add_argument("--out-dir", default="data/processed/05_adain_signature_augmented")
    parser.add_argument("--alpha", type=float, default=1.0, help="Interpolation strength toward the target scanner style.")
    parser.add_argument("--include-original", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(args.adain, map_location=device)
    model = ScannerAdaINAutoencoder(image_size=int(checkpoint["image_size"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    signatures = checkpoint["signatures"]

    df = pd.read_csv(args.source_csv)
    source_manufacturers = sorted(df["manufacturer"].dropna().unique().tolist())
    missing = [m for m in source_manufacturers if m not in signatures]
    if missing:
        raise RuntimeError(f"Missing AdaIN signatures for manufacturers: {missing}")

    rows = []
    if args.include_original:
        original = df.copy()
        original["image_path"] = ""
        original["synthetic_signature"] = ""
        original["synthetic_method"] = "original"
        rows.extend(original.to_dict(orient="records"))

    with torch.no_grad():
        for _, row in df.iterrows():
            x = load_center_slice(row["volume_path"], int(checkpoint["image_size"])).to(device)
            features = model.encode(x)
            source_manufacturer = row["manufacturer"]
            for target_manufacturer in source_manufacturers:
                if target_manufacturer == source_manufacturer:
                    continue
                style_mean = signatures[target_manufacturer]["mean"].to(device).unsqueeze(0)
                style_std = signatures[target_manufacturer]["std"].to(device).unsqueeze(0)
                styled_features = adaptive_instance_normalization(features, style_mean, style_std)
                if args.alpha < 1.0:
                    styled_features = (1.0 - args.alpha) * features + args.alpha * styled_features
                synthetic = model.decode(styled_features).squeeze().cpu().numpy().astype(np.float32)

                target_tag = safe_name(target_manufacturer)
                out_path = image_dir / f"{row['subject']}__{row['image_id']}__adain_{target_tag}.npy"
                np.save(out_path, synthetic)
                item = row.to_dict()
                item["image_path"] = str(out_path)
                item["synthetic_signature"] = target_manufacturer
                item["synthetic_source_manufacturer"] = source_manufacturer
                item["synthetic_method"] = "adain"
                rows.append(item)

    out_csv = out_dir / "source_train_adain_signature_augmented.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved AdaIN signature augmented CSV: {out_csv}")
    print(f"Input scans: {len(df)}")
    print(f"Output rows: {len(rows)}")


if __name__ == "__main__":
    main()
