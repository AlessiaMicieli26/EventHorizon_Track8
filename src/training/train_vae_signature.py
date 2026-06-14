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

from src.models.vae import ScannerSignatureVAE


class ScannerSliceDataset(Dataset):
    def __init__(self, csv_path: str, image_size: int = 256):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.image_size = image_size
        if "manufacturer" not in self.df.columns:
            raise RuntimeError("CSV must contain a manufacturer column.")
        if self.df["manufacturer"].nunique() < 2:
            raise RuntimeError("VAE signature training needs at least two manufacturers in the train CSV.")

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


def vae_loss(reconstruction, x, mu, logvar, beta: float):
    recon = F.mse_loss(reconstruction, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kld, recon, kld


@torch.no_grad()
def compute_signatures(model, loader, device):
    model.eval()
    latents: dict[str, list[torch.Tensor]] = {}
    for x, manufacturers in loader:
        mu, _ = model.encode(x.to(device))
        for manufacturer, z in zip(manufacturers, mu.cpu()):
            latents.setdefault(str(manufacturer), []).append(z)
    signatures = {}
    for manufacturer, vectors in latents.items():
        stacked = torch.stack(vectors)
        signatures[manufacturer] = {
            "mean": stacked.mean(dim=0),
            "std": stacked.std(dim=0, unbiased=False).clamp_min(1e-6),
            "n": stacked.shape[0],
        }
    return signatures


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a VAE and extract scanner signatures in latent space.")
    parser.add_argument("--train-csv", default="data/processed/01_domain_split/source_train.csv")
    parser.add_argument("--out-dir", default="experiments/outputs/vae_signature")
    parser.add_argument("--checkpoint-dir", default="experiments/checkpoints")
    parser.add_argument("--checkpoint-name", default="vae_signature.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--beta", type=float, default=1e-3)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ScannerSliceDataset(args.train_csv, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = ScannerSignatureVAE(latent_dim=args.latent_dim, image_size=args.image_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []

    for epoch in range(args.epochs):
        model.train()
        losses, recons, klds = [], [], []
        for x, _ in loader:
            x = x.to(device)
            reconstruction, mu, logvar = model(x)
            loss, recon, kld = vae_loss(reconstruction, x, mu, logvar, args.beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            recons.append(recon.item())
            klds.append(kld.item())
        row = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "reconstruction": float(np.mean(recons)),
            "kld": float(np.mean(klds)),
        }
        history.append(row)
        print(
            "epoch={epoch} loss={loss:.4f} recon={reconstruction:.4f} kld={kld:.4f}".format(
                **row
            )
        )

    signatures = compute_signatures(model, eval_loader, device)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "latent_dim": args.latent_dim,
        "image_size": args.image_size,
        "signatures": signatures,
    }
    torch.save(checkpoint, out_dir / "vae_signature.pt")
    torch.save({"signatures": signatures}, out_dir / "scanner_signatures.pt")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    manifest = {
        "train_csv": args.train_csv,
        "latent_dim": args.latent_dim,
        "image_size": args.image_size,
        "manufacturers": {
            manufacturer: {"n": int(info["n"])}
            for manufacturer, info in signatures.items()
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shutil.copy2(out_dir / "vae_signature.pt", checkpoint_dir / args.checkpoint_name)
    print(json.dumps(manifest, indent=2))
    print(f"Saved VAE checkpoint to {checkpoint_dir / args.checkpoint_name}")


if __name__ == "__main__":
    main()
