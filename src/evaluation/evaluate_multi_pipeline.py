import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))["target_test"]


def parse_model_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("Model specs must use NAME=PATH format.")
    name, path = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Model name cannot be empty.")
    return name, Path(path.strip())


def save_scatter(summary: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    ax.scatter(summary["accuracy"], summary["macro_f1"], s=95)
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
    ax.set_title("Target-Domain Comparison")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "multi_model_accuracy_f1.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple classifier metrics on the target test set.")
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model_spec,
        required=True,
        help="Model metric spec in NAME=path/to/metrics.json format. Repeat for each model.",
    )
    parser.add_argument("--out-dir", default="experiments/outputs/evaluation_multi")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()

    rows = []
    for name, path in args.model:
        metrics = load_metrics(path)
        rows.append(
            {
                "model": name,
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "metrics_path": str(path),
            }
        )
    summary = pd.DataFrame(rows)
    baseline_accuracy = summary.iloc[0]["accuracy"]
    baseline_f1 = summary.iloc[0]["macro_f1"]
    summary["delta_accuracy_vs_first"] = summary["accuracy"] - baseline_accuracy
    summary["delta_macro_f1_vs_first"] = summary["macro_f1"] - baseline_f1

    out_dir = Path(args.out_dir)
    figures_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(out_dir / "summary.csv", index=False)
    summary.to_json(out_dir / "summary.json", orient="records", indent=2)
    lines = [
        "| model | accuracy | macro_f1 | delta_accuracy_vs_first | delta_macro_f1_vs_first |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            "| {model} | {accuracy:.4f} | {macro_f1:.4f} | {delta_accuracy_vs_first:.4f} | {delta_macro_f1_vs_first:.4f} |".format(
                **row
            )
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_scatter(summary, figures_dir)

    print(summary.to_string(index=False))
    print(f"Saved multi-model evaluation to {out_dir}")
    print(f"Saved multi-model scatter to {figures_dir / 'multi_model_accuracy_f1.png'}")


if __name__ == "__main__":
    main()
