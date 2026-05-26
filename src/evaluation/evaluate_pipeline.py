import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def load_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def row_from_metrics(name: str, path: Path) -> dict:
    metrics = load_metrics(path)["target_test"]
    return {
        "model": name,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "metrics_path": str(path),
    }


def save_metric_scatter(summary: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    ax.scatter(summary["accuracy"], summary["macro_f1"], s=90, color=["#1f77b4", "#d62728"])
    for _, row in summary.iterrows():
        ax.annotate(
            row["model"],
            (row["accuracy"], row["macro_f1"]),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel("Target accuracy")
    ax.set_ylabel("Target macro-F1")
    ax.set_title("Target-Domain Performance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "metric_scatter_accuracy_f1.png")
    plt.close(fig)


def save_training_curves(history_paths: list[tuple[str, Path]], figures_dir: Path) -> None:
    histories = []
    for name, path in history_paths:
        if path.exists():
            history = pd.read_csv(path)
            history["model"] = name
            histories.append(history)
    if not histories:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=160)
    for history in histories:
        label = history["model"].iloc[0]
        epoch = history["epoch"].to_numpy()
        axes[0].plot(epoch, history["loss"].to_numpy(), marker="o", linewidth=1.6, markersize=3, label=label)
        if "val_acc" in history:
            axes[1].plot(epoch, history["val_acc"].to_numpy(), marker="o", linewidth=1.6, markersize=3, label=label)
        if "val_f1" in history:
            axes[2].plot(epoch, history["val_f1"].to_numpy(), marker="o", linewidth=1.6, markersize=3, label=label)

    axes[0].set_title("Training Loss")
    axes[1].set_title("Validation Accuracy")
    axes[2].set_title("Validation Macro-F1")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Accuracy")
    axes[2].set_ylabel("Macro-F1")
    fig.tight_layout()
    fig.savefig(figures_dir / "classifier_temporal_metrics.png")
    plt.close(fig)


def save_cyclegan_curves(history_path: Path, figures_dir: Path) -> None:
    if not history_path.exists():
        return
    history = pd.read_csv(history_path)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
    epoch = history["epoch"].to_numpy()
    ax.plot(epoch, history["generator"].to_numpy(), marker="o", linewidth=1.6, markersize=3, label="generator")
    ax.plot(epoch, history["discriminator"].to_numpy(), marker="o", linewidth=1.6, markersize=3, label="discriminator")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("CycleGAN Temporal Losses")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "cyclegan_temporal_losses.png")
    plt.close(fig)


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    vmin, vmax = float(image.min()), float(image.max())
    if vmax <= vmin:
        return np.zeros_like(image)
    return (image - vmin) / (vmax - vmin)


def load_center_slice(volume_path: str) -> np.ndarray:
    volume = np.load(volume_path, mmap_mode="r")
    return np.asarray(volume[:, :, volume.shape[2] // 2], dtype=np.float32)


def resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if image.shape == reference.shape:
        return image
    tensor = torch.from_numpy(image[None, None, ...].copy())
    resized = F.interpolate(tensor, size=reference.shape, mode="bilinear", align_corners=False)
    return resized.squeeze().numpy()


def save_qualitative_preview(source_csv: Path, translated_csv: Path, figures_dir: Path, n: int = 6) -> None:
    if not source_csv.exists() or not translated_csv.exists():
        return
    source = pd.read_csv(source_csv).head(n)
    translated = pd.read_csv(translated_csv).head(n)
    n = min(len(source), len(translated), n)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 3, figsize=(8, 2.3 * n), dpi=160)
    if n == 1:
        axes = np.expand_dims(axes, axis=0)
    for i in range(n):
        original = normalize_image(load_center_slice(source.iloc[i]["volume_path"]))
        translated_img = normalize_image(np.load(translated.iloc[i]["image_path"]))
        original = resize_like(original, translated_img)
        diff = np.abs(translated_img - original)

        axes[i, 0].imshow(original, cmap="gray")
        axes[i, 0].set_title("Original source")
        axes[i, 1].imshow(translated_img, cmap="gray")
        axes[i, 1].set_title("CycleGAN translated")
        axes[i, 2].imshow(diff, cmap="magma")
        axes[i, 2].set_title("Absolute difference")
        for ax in axes[i]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(figures_dir / "qualitative_translation_previews.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare target-domain metrics for the ADNI adaptation pipeline.")
    parser.add_argument(
        "--baseline",
        default="experiments/outputs/classifier_source_only/metrics.json",
        help="Source-only classifier metrics JSON.",
    )
    parser.add_argument(
        "--augmented",
        default="experiments/outputs/classifier_cyclegan_augmented/metrics.json",
        help="CycleGAN-augmented classifier metrics JSON.",
    )
    parser.add_argument("--out-dir", default="experiments/outputs/evaluation")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--baseline-history", default="experiments/logs/classifier_source_only_history.csv")
    parser.add_argument("--augmented-history", default="experiments/logs/classifier_cyclegan_augmented_history.csv")
    parser.add_argument("--cyclegan-history", default="experiments/logs/cyclegan_scanner_history.csv")
    parser.add_argument("--source-csv", default="data/processed/01_domain_split/source_train.csv")
    parser.add_argument("--translated-csv", default="data/processed/02_translated_source/translated_source_train.csv")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        row_from_metrics("source_only", Path(args.baseline)),
        row_from_metrics("cyclegan_augmented", Path(args.augmented)),
    ]
    summary = pd.DataFrame(rows)
    summary["delta_accuracy_vs_source_only"] = summary["accuracy"] - summary.loc[0, "accuracy"]
    summary["delta_macro_f1_vs_source_only"] = summary["macro_f1"] - summary.loc[0, "macro_f1"]

    summary_csv = out_dir / "summary.csv"
    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"
    summary.to_csv(summary_csv, index=False)
    summary.to_json(summary_json, orient="records", indent=2)
    lines = [
        "| model | accuracy | macro_f1 | delta_accuracy_vs_source_only | delta_macro_f1_vs_source_only |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            "| {model} | {accuracy:.4f} | {macro_f1:.4f} | {delta_accuracy_vs_source_only:.4f} | {delta_macro_f1_vs_source_only:.4f} |".format(
                **row
            )
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_metric_scatter(summary, figures_dir)
    save_training_curves(
        [
            ("source_only", Path(args.baseline_history)),
            ("cyclegan_augmented", Path(args.augmented_history)),
        ],
        figures_dir,
    )
    save_cyclegan_curves(Path(args.cyclegan_history), figures_dir)
    save_qualitative_preview(Path(args.source_csv), Path(args.translated_csv), figures_dir)

    print(summary.to_string(index=False))
    print(f"Saved evaluation summary to {out_dir}")
    print(f"Saved evaluation figures to {figures_dir}")


if __name__ == "__main__":
    main()
