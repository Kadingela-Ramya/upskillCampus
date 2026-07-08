"""
split_dataset.py  (FarmEye-AI)

The raw dataset comes as one flat folder: every image and its matching
YOLO .txt label sitting side by side, no train/val/test division. This
script does that split for us - it pairs each image with its label file,
shuffles the pairs (so we're not just taking images in whatever order
they happen to be listed), and copies them into the
dataset/images/{train,val,test} and dataset/labels/{train,val,test}
layout the rest of the project expects.

Copying rather than moving is deliberate - if something goes wrong
partway through, the original flat folder is untouched and we can just
rerun this.

Run it like:
    python split_dataset.py --source "Project5_Ag_Crop and weed detection/agri_data/data"

Default split is 70% train / 20% val / 10% test - change with --train-pct
and --val-pct if you want something different (test gets whatever's left).
"""

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png")


def find_image_label_pairs(source_dir: Path):
    """Walks the flat source folder and matches each image to its .txt label
    by filename (agri_0_8322.jpeg <-> agri_0_8322.txt). Images with no
    matching label are reported and skipped rather than silently dropped."""
    pairs = []
    missing_labels = []

    for image_path in sorted(source_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = source_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))
        else:
            missing_labels.append(image_path.name)

    return pairs, missing_labels


def copy_pair(image_path, label_path, images_out, labels_out):
    shutil.copy2(image_path, images_out / image_path.name)
    shutil.copy2(label_path, labels_out / label_path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="folder containing the flat images+labels")
    parser.add_argument("--dest", default="dataset", help="output dataset root")
    parser.add_argument("--train-pct", type=float, default=0.70)
    parser.add_argument("--val-pct", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42, help="fixed seed so the split is reproducible")
    args = parser.parse_args()

    source_dir = Path(args.source)
    dest_root = Path(args.dest)

    if not source_dir.exists():
        print(f"Source folder not found: {source_dir}")
        return

    pairs, missing_labels = find_image_label_pairs(source_dir)
    print(f"Found {len(pairs)} image+label pairs.")
    if missing_labels:
        print(f"Warning: {len(missing_labels)} images had no matching .txt label and were skipped:")
        for name in missing_labels[:10]:
            print(f"    {name}")
        if len(missing_labels) > 10:
            print(f"    ...and {len(missing_labels) - 10} more")

    random.seed(args.seed)
    random.shuffle(pairs)

    total = len(pairs)
    train_end = int(total * args.train_pct)
    val_end = train_end + int(total * args.val_pct)

    splits = {
        "train": pairs[:train_end],
        "val": pairs[train_end:val_end],
        "test": pairs[val_end:],
    }

    for split_name, split_pairs in splits.items():
        images_out = dest_root / "images" / split_name
        labels_out = dest_root / "labels" / split_name
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        for image_path, label_path in split_pairs:
            copy_pair(image_path, label_path, images_out, labels_out)

        print(f"{split_name}: {len(split_pairs)} images -> {images_out}")

    print("\nDone. Dataset is ready at:", dest_root.resolve())


if __name__ == "__main__":
    main()
