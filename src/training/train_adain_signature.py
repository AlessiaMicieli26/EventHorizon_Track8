import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.models.adain import ScannerAdaINAutoencoder, channel_mean_std


class ScannerSliceDataset(Dataset):
    def __init__(self, csv_path: str, image_size: int = 256):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.image_size = image_size
        if "manufacturer" not in self.df.columns:
            raise RuntimeError("CSV must contain a manufacturer column.")
        if self.df["manufacturer"].nunique() < 2:
            raise RuntimeError("AdaIN signature training needs at least two manufacturers in the train CSV.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        volume = np.load(row["volume_path"], mmap_mode="r")
        image = np.asarray(volume[:, :, volume.shape[2] // 2], dtype=np.float32)
        image = torch.from_numpy(image[None, ...].copy())
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return image, row["manufacturer"]


@torch.no_grad()
def compute_signatures(model, loader, device):
    model.eval()
    stats: dict[str, dict[str, list[torch.Tensor]]] = {}
    for x, manufacturers in loader:
        features = model.encode(x.to(device))
        mean, std = channel_mean_std(features)
        for manufacturer, item_mean, item_std in zip(manufacturers, mean.cpu(), std.cpu()):
            entry = stats.setdefault(str(manufacturer), {"mean": [], "std": []})
            entry["mean"].append(item_mean)
            entry["std"].append(item_std)

    signatures = {}
    for manufacturer, values in stats.items():
        means = torch.stack(values["mean"])
        stds = torch.stack(values["std"])
        signatures[manufacturer] = {
            "mean": means.mean(dim=0),
            "std": stds.mean(dim=0).clamp_min(1e-6),
            "n": means.shape[0],
        }
    return signatures


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an AdaIN autoencoder and extract scanner style signatures.")
    parser.add_argument("--train-csv", default="data/processed/01_domain_split/source_train.csv")
    parser.add_argument("--out-dir", default="experiments/outputs/adain_signature")
    parser.add_argument("--checkpoint-dir", default="experiments/checkpoints")
    parser.add_argument("--checkpoint-name", default="adain_signature.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ScannerSliceDataset(args.train_csv, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = ScannerAdaINAutoencoder(image_size=args.image_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for x, _ in loader:
            x = x.to(device)
            reconstruction = model(x)
            loss = F.mse_loss(reconstruction, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses))}
        history.append(row)
        print("epoch={epoch} loss={loss:.4f}".format(**row))

    signatures = compute_signatures(model, eval_loader, device)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "image_size": args.image_size,
        "signatures": signatures,
    }
    torch.save(checkpoint, out_dir / "adain_signature.pt")
    torch.save({"signatures": signatures}, out_dir / "scanner_signatures.pt")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    manifest = {
        "train_csv": args.train_csv,
        "image_size": args.image_size,
        "manufacturers": {
            manufacturer: {"n": int(info["n"])}
            for manufacturer, info in signatures.items()
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shutil.copy2(out_dir / "adain_signature.pt", checkpoint_dir / args.checkpoint_name)
    print(json.dumps(manifest, indent=2))
    print(f"Saved AdaIN checkpoint to {checkpoint_dir / args.checkpoint_name}")


if __name__ == "__main__":
    main()
