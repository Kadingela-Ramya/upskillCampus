"""
verify_labels.py  (FarmEye-AI)

Before spending an hour training on a dataset, it's worth spending two
minutes looking at whether the bounding boxes actually line up with the
plants they claim to describe. This script pulls a handful of random
images, redraws their YOLO labels on top, and dumps them in a folder
so you can eyeball them.

Nothing fancy - if a box is way off, or a crop is labeled as a weed,
you'll catch it here instead of during training.

Run it like:
    python verify_labels.py --images dataset/images/train --labels dataset/labels/train
"""

import argparse
import random
from pathlib import Path

import cv2

# Keeping colors distinct and high-contrast so mistakes are obvious at a glance
BOX_COLOR = {"crop": (60, 200, 60), "weed": (0, 140, 255)}   # green / orange, BGR
CLASS_LOOKUP = {0: "crop", 1: "weed"}


def read_yolo_labels(label_file: Path):
    """Parses a single YOLO label .txt file into a list of (class_id, xc, yc, w, h)."""
    if not label_file.exists():
        return []

    boxes = []
    for raw_line in label_file.read_text().splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        cls_id, xc, yc, w, h = raw_line.split()
        boxes.append((int(cls_id), float(xc), float(yc), float(w), float(h)))
    return boxes


def normalized_box_to_pixels(xc, yc, w, h, img_width, img_height):
    """YOLO stores boxes as fractions of image size, centered on the box.
    Convert that to actual top-left / bottom-right pixel coordinates."""
    box_w = w * img_width
    box_h = h * img_height
    center_x = xc * img_width
    center_y = yc * img_height

    left = int(center_x - box_w / 2)
    top = int(center_y - box_h / 2)
    right = int(center_x + box_w / 2)
    bottom = int(center_y + box_h / 2)
    return left, top, right, bottom


def annotate_image(image_path: Path, label_path: Path, save_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not open {image_path}, skipping.")
        return

    height, width = image.shape[:2]
    boxes = read_yolo_labels(label_path)

    if not boxes:
        print(f"No label file found for {image_path.name} - saving unmarked.")

    for cls_id, xc, yc, w, h in boxes:
        class_name = CLASS_LOOKUP.get(cls_id, f"class_{cls_id}")
        color = BOX_COLOR.get(class_name, (255, 255, 255))
        left, top, right, bottom = normalized_box_to_pixels(xc, yc, w, h, width, height)

        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        cv2.putText(image, class_name, (left, max(top - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite(str(save_path), image)


def main():
    parser = argparse.ArgumentParser(description="Redraw YOLO labels on sample images for a visual check.")
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--samples", type=int, default=10, help="how many random images to check")
    parser.add_argument("--out", default="label_check")
    args = parser.parse_args()

    image_dir = Path(args.images)
    label_dir = Path(args.labels)
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    all_images = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not all_images:
        print(f"No images found in {image_dir}")
        return

    chosen = random.sample(all_images, min(args.samples, len(all_images)))

    for img_path in chosen:
        matching_label = label_dir / f"{img_path.stem}.txt"
        output_path = out_dir / img_path.name
        annotate_image(img_path, matching_label, output_path)

    print(f"Checked {len(chosen)} images. Open '{out_dir}/' and look them over before training.")


if __name__ == "__main__":
    main()
