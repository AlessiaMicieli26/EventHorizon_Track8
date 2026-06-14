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
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset

from src.models.classifier import SmallMRIClassifier


class MRISliceDataset(Dataset):
    def __init__(self, csv_path: str, label_to_idx: dict[str, int] | None = None, image_size: int = 256):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        labels = sorted(self.df["disease_label"].unique()) if label_to_idx is None else list(label_to_idx)
        self.label_to_idx = label_to_idx or {label: idx for idx, label in enumerate(labels)}
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        if "image_path" in row and isinstance(row["image_path"], str) and row["image_path"]:
            image = np.load(row["image_path"]).astype(np.float32)
        else:
            volume = np.load(row["volume_path"], mmap_mode="r")
            image = np.asarray(volume[:, :, volume.shape[2] // 2], dtype=np.float32)
        image = torch.from_numpy(np.expand_dims(image, axis=0).copy())
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        y = self.label_to_idx[row["disease_label"]]
        return image, torch.tensor(y, dtype=torch.long)


def train_epoch(model, loader, optimizer, device):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = criterion(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        pred = model(x.to(device)).argmax(dim=1).cpu().numpy().tolist()
        y_true.extend(y.numpy().tolist())
        y_pred.extend(pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MRI disease classifier and evaluate target-domain generalization.")
    parser.add_argument("--train-csv", default="data/processed/01_domain_split/source_train.csv")
    parser.add_argument("--val-csv", default="data/processed/01_domain_split/source_val.csv")
    parser.add_argument("--test-csv", default="data/processed/01_domain_split/target_test.csv")
    parser.add_argument("--out-dir", default="experiments/outputs/classifier_source_only")
    parser.add_argument(
        "--checkpoint-dir",
        default="experiments/checkpoints",
        help="Directory where final model checkpoints are also copied.",
    )
    parser.add_argument("--checkpoint-name", default=None, help="Checkpoint filename inside --checkpoint-dir.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = MRISliceDataset(args.train_csv)
    val_ds = MRISliceDataset(args.val_csv, train_ds.label_to_idx)
    test_ds = MRISliceDataset(args.test_csv, train_ds.label_to_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SmallMRIClassifier(n_classes=len(train_ds.label_to_idx)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val = -1.0

    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        print(f"epoch={epoch + 1} loss={loss:.4f} val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['macro_f1']:.4f}")
        if val_metrics["macro_f1"] > best_val:
            best_val = val_metrics["macro_f1"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")

    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    result = {"label_to_idx": train_ds.label_to_idx, "target_test": test_metrics}
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    checkpoint_name = args.checkpoint_name or f"{out_dir.name}_best_model.pt"
    shutil.copy2(out_dir / "best_model.pt", checkpoint_dir / checkpoint_name)
    (checkpoint_dir / f"{Path(checkpoint_name).stem}_labels.json").write_text(
        json.dumps(train_ds.label_to_idx, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    print(f"Saved model checkpoint to {checkpoint_dir / checkpoint_name}")


if __name__ == "__main__":
    main()
