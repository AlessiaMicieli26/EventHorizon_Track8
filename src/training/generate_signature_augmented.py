import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.models.vae import ScannerSignatureVAE


def load_center_slice(path: str, image_size: int) -> torch.Tensor:
    volume = np.load(path, mmap_mode="r")
    image = np.asarray(volume[:, :, volume.shape[2] // 2], dtype=np.float32)
    x = torch.from_numpy(image[None, None, ...].copy())
    if x.shape[-2:] != (image_size, image_size):
        x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return x.clamp(0.0, 1.0)


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create synthetic scanner-signature slices by adding VAE latent signature vectors to source images."
    )
    parser.add_argument("--source-csv", default="data/processed/01_domain_split/source_train.csv")
    parser.add_argument("--vae", default="experiments/checkpoints/vae_signature.pt")
    parser.add_argument("--out-dir", default="data/processed/03_vae_signature_augmented")
    parser.add_argument("--alpha", type=float, default=1.0, help="Strength of target-source signature transfer.")
    parser.add_argument("--noise-scale", type=float, default=0.25, help="Latent std multiplier for synthetic signatures.")
    parser.add_argument(
        "--generation-mode",
        choices=["residual", "decode"],
        default="residual",
        help=(
            "residual preserves anatomy by adding the decoded scanner-style delta to the original slice; "
            "decode saves the raw shifted VAE decoder output."
        ),
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=0.6,
        help="Multiplier for the decoded scanner-style residual when generation-mode=residual.",
    )
    parser.add_argument("--copies-per-target", type=int, default=1)
    parser.add_argument("--include-original", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(args.vae, map_location=device)
    model = ScannerSignatureVAE(
        latent_dim=int(checkpoint["latent_dim"]),
        image_size=int(checkpoint["image_size"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    signatures = checkpoint["signatures"]

    df = pd.read_csv(args.source_csv)
    source_manufacturers = sorted(df["manufacturer"].dropna().unique().tolist())
    missing = [m for m in source_manufacturers if m not in signatures]
    if missing:
        raise RuntimeError(f"Missing VAE signatures for manufacturers: {missing}")

    rows = []
    if args.include_original:
        original = df.copy()
        original["image_path"] = ""
        original["synthetic_signature"] = ""
        rows.extend(original.to_dict(orient="records"))

    with torch.no_grad():
        for _, row in df.iterrows():
            x = load_center_slice(row["volume_path"], checkpoint["image_size"]).to(device)
            mu, _ = model.encode(x)
            decoded_source = model.decode(mu)
            source_manufacturer = row["manufacturer"]
            source_mean = signatures[source_manufacturer]["mean"].to(device)
            for target_manufacturer in source_manufacturers:
                if target_manufacturer == source_manufacturer:
                    continue
                target_mean = signatures[target_manufacturer]["mean"].to(device)
                target_std = signatures[target_manufacturer]["std"].to(device)
                direction = target_mean - source_mean
                for copy_idx in range(args.copies_per_target):
                    noise = torch.randn_like(mu) * target_std.unsqueeze(0) * args.noise_scale
                    z = mu + args.alpha * direction.unsqueeze(0) + noise
                    decoded_shifted = model.decode(z)
                    if args.generation_mode == "residual":
                        scanner_delta = decoded_shifted - decoded_source
                        synthetic_tensor = (x + args.residual_scale * scanner_delta).clamp(0.0, 1.0)
                    else:
                        synthetic_tensor = decoded_shifted.clamp(0.0, 1.0)
                    synthetic = synthetic_tensor.squeeze().cpu().numpy().astype(np.float32)
                    target_tag = safe_name(target_manufacturer)
                    out_path = image_dir / f"{row['subject']}__{row['image_id']}__sig_{target_tag}__{copy_idx}.npy"
                    np.save(out_path, synthetic)
                    item = row.to_dict()
                    item["image_path"] = str(out_path)
                    item["synthetic_signature"] = target_manufacturer
                    item["synthetic_source_manufacturer"] = source_manufacturer
                    rows.append(item)

    out_csv = out_dir / "source_train_vae_signature_augmented.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved VAE signature augmented CSV: {out_csv}")
    print(f"Input scans: {len(df)}")
    print(f"Output rows: {len(rows)}")


if __name__ == "__main__":
    main()
