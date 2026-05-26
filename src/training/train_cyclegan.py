import argparse
import itertools
import random
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.models.cyclegan import Discriminator, Generator


class UnpairedSliceDataset(Dataset):
    def __init__(self, source_csv: str, target_csv: str, image_size: int = 256):
        self.source = pd.read_csv(source_csv)["volume_path"].tolist()
        self.target = pd.read_csv(target_csv)["volume_path"].tolist()
        self.image_size = image_size
        if not self.source or not self.target:
            raise RuntimeError("Both source and target CSVs must contain at least one scan.")

    def __len__(self):
        return max(len(self.source), len(self.target))

    def load_center_slice(self, path: str):
        volume = np.load(path, mmap_mode="r")
        image = np.asarray(volume[:, :, volume.shape[2] // 2], dtype=np.float32)
        image = torch.from_numpy(image[None, ...].copy())
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return image

    def __getitem__(self, idx):
        source_path = self.source[idx % len(self.source)]
        target_path = self.target[random.randrange(len(self.target))]
        return self.load_center_slice(source_path), self.load_center_slice(target_path)


def gan_loss(prediction, real: bool):
    target = torch.ones_like(prediction) if real else torch.zeros_like(prediction)
    return F.mse_loss(prediction, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CycleGAN between source and target MRI scanner domains.")
    parser.add_argument("--source-csv", default="data/processed/01_domain_split/source_train.csv")
    parser.add_argument("--target-csv", default="data/processed/01_domain_split/target_adapt_unlabeled.csv")
    parser.add_argument("--out-dir", default="experiments/outputs/cyclegan_scanner")
    parser.add_argument(
        "--checkpoint-dir",
        default="experiments/checkpoints",
        help="Directory where final generator checkpoints are also copied.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-cycle", type=float, default=10.0)
    parser.add_argument("--lambda-identity", type=float, default=2.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loader = DataLoader(
        UnpairedSliceDataset(args.source_csv, args.target_csv),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    g_st, g_ts = Generator().to(device), Generator().to(device)
    d_s, d_t = Discriminator().to(device), Discriminator().to(device)

    opt_g = torch.optim.Adam(itertools.chain(g_st.parameters(), g_ts.parameters()), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(itertools.chain(d_s.parameters(), d_t.parameters()), lr=args.lr, betas=(0.5, 0.999))

    for epoch in range(args.epochs):
        for source, target in loader:
            source, target = source.to(device), target.to(device)
            fake_target = g_st(source)
            fake_source = g_ts(target)
            rec_source = g_ts(fake_target)
            rec_target = g_st(fake_source)

            loss_g = (
                gan_loss(d_t(fake_target), True)
                + gan_loss(d_s(fake_source), True)
                + args.lambda_cycle * (F.l1_loss(rec_source, source) + F.l1_loss(rec_target, target))
                + args.lambda_identity * (F.l1_loss(g_st(target), target) + F.l1_loss(g_ts(source), source))
            )
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            loss_d = (
                gan_loss(d_s(source), True)
                + gan_loss(d_s(fake_source.detach()), False)
                + gan_loss(d_t(target), True)
                + gan_loss(d_t(fake_target.detach()), False)
            )
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

        print(f"epoch={epoch + 1} generator={loss_g.item():.4f} discriminator={loss_d.item():.4f}")

    torch.save(g_st.state_dict(), out_dir / "generator_source_to_target.pt")
    torch.save(g_ts.state_dict(), out_dir / "generator_target_to_source.pt")
    shutil.copy2(out_dir / "generator_source_to_target.pt", checkpoint_dir / "cyclegan_generator_source_to_target.pt")
    shutil.copy2(out_dir / "generator_target_to_source.pt", checkpoint_dir / "cyclegan_generator_target_to_source.pt")
    print(f"Saved CycleGAN generators to {out_dir}")
    print(f"Copied CycleGAN generators to {checkpoint_dir}")


if __name__ == "__main__":
    main()
