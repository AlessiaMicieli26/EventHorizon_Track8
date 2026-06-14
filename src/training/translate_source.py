import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.models.cyclegan import Generator


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate source-domain MRI slices to target scanner style.")
    parser.add_argument("--source-csv", default="data/processed/01_domain_split/source_train.csv")
    parser.add_argument("--generator", default="experiments/outputs/cyclegan_scanner/generator_source_to_target.pt")
    parser.add_argument("--out-dir", default="data/processed/02_translated_source")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Generator().to(device)
    model.load_state_dict(torch.load(args.generator, map_location=device))
    model.eval()

    rows = []
    df = pd.read_csv(args.source_csv)
    for _, row in df.iterrows():
        volume = np.load(row["volume_path"], mmap_mode="r")
        image = np.asarray(volume[:, :, volume.shape[2] // 2], dtype=np.float32)
        x = torch.from_numpy(image[None, None, ...].copy()).to(device)
        if x.shape[-2:] != (256, 256):
            x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        with torch.no_grad():
            translated = model(x).squeeze().cpu().numpy().astype(np.float32)

        out_path = image_dir / f"{row['subject']}__{row['image_id']}__translated.npy"
        np.save(out_path, translated)
        item = row.to_dict()
        item["image_path"] = str(out_path)
        rows.append(item)

    out_csv = out_dir / "translated_source_train.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved translated source CSV: {out_csv}")


if __name__ == "__main__":
    main()
