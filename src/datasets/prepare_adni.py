import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def normalize_image_id(value: object) -> str:
    text = str(value).strip()
    return text if text.upper().startswith("I") else f"I{text}"


def infer_modality(description: str) -> str:
    text = (description or "").upper()
    if "FLAIR" in text:
        return "FLAIR"
    if "T2" in text:
        return "T2"
    if "T1" in text or "MPR" in text:
        return "T1"
    return "UNKNOWN"


def field_bin(field_strength: Optional[float]) -> Optional[str]:
    if field_strength is None or np.isnan(field_strength):
        return None
    return "1.5T" if field_strength < 2.25 else "3T"


def disease_label(group: str, mode: str) -> Optional[str]:
    group = str(group).upper()
    if mode == "cn_vs_ad":
        if group == "CN":
            return "healthy"
        if group == "AD":
            return "disease"
        return None
    if mode == "cn_vs_non_cn":
        return "healthy" if group == "CN" else "disease"
    raise ValueError(f"Unsupported label mode: {mode}")


def minmax_normalize(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32)
    vmin = float(np.min(volume))
    vmax = float(np.max(volume))
    if vmax <= vmin:
        return np.zeros_like(volume, dtype=np.float32)
    return (volume - vmin) / (vmax - vmin)


def image_ids_in_path(path: Path) -> set[str]:
    return {match.upper() for match in re.findall(r"I\d+", str(path))}


def build_file_index(root: Path, patterns: tuple[str, ...]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for pattern in patterns:
        for path in root.rglob(pattern):
            for image_id in image_ids_in_path(path):
                index.setdefault(image_id, []).append(path)
    return index


def find_indexed_path(image_id: str, index: dict[str, list[Path]], largest: bool = False) -> Optional[Path]:
    candidates = index.get(image_id.upper(), [])
    if not candidates:
        return None
    if largest:
        return max(candidates, key=lambda p: p.stat().st_size)
    return sorted(candidates)[0]


def parse_ida_metadata(xml_path: Optional[Path]) -> dict:
    meta = {
        "manufacturer": None,
        "scanner_model": None,
        "field_strength": None,
        "series_uid": None,
    }
    if xml_path is None:
        return meta

    root = ET.parse(xml_path).getroot()
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag == "seriesIdentifier" and elem.text:
            meta["series_uid"] = elem.text.strip()
        if tag != "protocol":
            continue
        term = elem.attrib.get("term", "").strip().lower()
        value = (elem.text or "").strip()
        if term == "manufacturer":
            meta["manufacturer"] = value.upper()
        elif term == "mfg model":
            meta["scanner_model"] = value
        elif term == "field strength":
            try:
                meta["field_strength"] = float(value)
            except ValueError:
                meta["field_strength"] = None
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare ADNI T1 MRI volumes and attach scanner/domain metadata."
    )
    parser.add_argument(
        "--csv",
        required=True,
        action="append",
        nargs="+",
        help="ADNI image collection CSV. Can be passed multiple times or with multiple paths.",
    )
    parser.add_argument("--nifti-root", required=True, help="Folder containing extracted .nii/.nii.gz files.")
    parser.add_argument("--metadata-root", required=True, help="Folder containing extracted IDA XML metadata.")
    parser.add_argument("--out-dir", default="data/processed/00_prepared")
    parser.add_argument("--field", choices=["auto", "1.5T", "3T"], default="auto")
    parser.add_argument("--label-mode", choices=["cn_vs_ad", "cn_vs_non_cn"], default="cn_vs_ad")
    parser.add_argument("--dry-run", action="store_true", help="Only report counts; do not save volumes.")
    args = parser.parse_args()

    csv_paths = [Path(path) for group in args.csv for path in group]
    nifti_root = Path(args.nifti_root)
    metadata_root = Path(args.metadata_root)
    out_dir = Path(args.out_dir)
    volume_dir = out_dir / "volumes"
    metadata_dir = out_dir / "metadata"

    frames = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        frame["source_csv"] = str(csv_path)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df["normalized_image_id"] = df["Image Data ID"].map(normalize_image_id)
    before = len(df)
    df = df.drop_duplicates(subset=["normalized_image_id"], keep="first").reset_index(drop=True)
    print(f"Loaded {len(csv_paths)} CSV file(s), {before} rows, {len(df)} unique image IDs")
    nifti_index = build_file_index(nifti_root, ("*.nii", "*.nii.gz"))
    metadata_index = build_file_index(metadata_root, ("*.xml",))
    print(f"Indexed {sum(len(v) for v in nifti_index.values())} NIfTI matches for {len(nifti_index)} image IDs")
    print(f"Indexed {sum(len(v) for v in metadata_index.values())} XML matches for {len(metadata_index)} image IDs")

    records = []
    for _, row in df.iterrows():
        image_id = row["normalized_image_id"]
        modality = infer_modality(str(row.get("Description", "")))
        if modality != "T1":
            continue

        label = disease_label(str(row["Group"]), args.label_mode)
        if label is None:
            continue

        nifti_path = find_indexed_path(image_id, nifti_index)
        xml_path = find_indexed_path(image_id, metadata_index, largest=True)
        scanner = parse_ida_metadata(xml_path)
        bin_name = field_bin(scanner["field_strength"])
        if nifti_path is None or scanner["manufacturer"] is None or bin_name is None:
            continue

        records.append(
            {
                "subject": str(row["Subject"]),
                "group": str(row["Group"]),
                "disease_label": label,
                "image_id": image_id,
                "modality": modality,
                "description": str(row.get("Description", "")),
                "acq_date": str(row.get("Acq Date", "")),
                "manufacturer": scanner["manufacturer"],
                "scanner_model": scanner["scanner_model"],
                "field_strength": scanner["field_strength"],
                "field_bin": bin_name,
                "series_uid": scanner["series_uid"],
                "source_nifti": str(nifti_path),
                "source_xml": str(xml_path) if xml_path else "",
                "source_csv": str(row.get("source_csv", "")),
            }
        )

    index = pd.DataFrame(records)
    if index.empty:
        raise RuntimeError("No usable ADNI T1 records found. Check CSV, NIfTI root, and metadata root.")

    chosen_field = args.field
    if chosen_field == "auto":
        chosen_field = Counter(index["field_bin"]).most_common(1)[0][0]
    index = index[index["field_bin"] == chosen_field].copy()

    print("Prepared metadata counts:")
    print(index.groupby(["field_bin", "manufacturer", "disease_label"]).size())
    print(f"Selected field strength: {chosen_field}")
    if args.dry_run:
        return

    import nibabel as nib

    volume_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    output_rows = []
    for _, row in index.iterrows():
        img = nib.load(row["source_nifti"])
        volume = minmax_normalize(np.asarray(img.get_fdata()))
        stem = f"{row['subject']}__{row['image_id']}__{row['manufacturer']}__{row['field_bin']}"
        volume_path = volume_dir / f"{stem}.npy"
        np.save(volume_path, volume.astype(np.float32))

        item = row.to_dict()
        item["volume_path"] = str(volume_path)
        with open(metadata_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(item, f, indent=2)
        output_rows.append(item)

    out_csv = out_dir / "index_prepared.csv"
    pd.DataFrame(output_rows).to_csv(out_csv, index=False)
    print(f"Saved {len(output_rows)} prepared volumes to {out_csv}")


if __name__ == "__main__":
    main()
