"""
analyze_field.py  (FarmEye-AI)

This is the part of the project that actually matters to a farmer looking
at the output: given a photo of a field, how much of it is weed?

For every image it:
  1. Runs the trained FarmEye-AI model to find crop/weed boxes
  2. Counts how many of each
  3. Estimates weed coverage as a percentage of the image area taken up
     by weed boxes (a rough proxy for "how much of this patch is weed",
     not a pixel-perfect segmentation - good enough to flag problem areas)
  4. Draws the boxes back onto the image along with a coverage summary
  5. Writes a plain-text report across all processed images

Run it like:
    python analyze_field.py --source dataset/images/test
    python analyze_field.py --source my_field_photo.jpg --weights runs/farmeye/farmeye_train/weights/best.pt
"""

import argparse
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

BOX_COLOR = {"crop": (60, 200, 60), "weed": (0, 140, 255)}  # BGR: green / orange


def box_area(xyxy):
    """Area of a single box given (x1, y1, x2, y2) pixel coordinates."""
    x1, y1, x2, y2 = xyxy
    return max(0, x2 - x1) * max(0, y2 - y1)


def summarize_image(result):
    """
    Pulls out per-class counts and a rough weed-coverage percentage from
    one Ultralytics prediction result.

    Coverage here = (sum of weed box areas) / (total image area) * 100.
    This is a bounding-box estimate, not true pixel segmentation, so it
    will slightly overestimate coverage when boxes overlap - worth knowing,
    but it's a reasonable and fast stand-in for "how much weed is here".
    """
    img_height, img_width = result.orig_shape
    total_area = img_width * img_height

    class_counts = Counter()
    weed_area_sum = 0

    for box in result.boxes:
        cls_name = result.names[int(box.cls[0])]
        class_counts[cls_name] += 1

        if cls_name == "weed":
            xyxy = box.xyxy[0].tolist()
            weed_area_sum += box_area(xyxy)

    coverage_pct = (weed_area_sum / total_area) * 100 if total_area else 0
    return class_counts, coverage_pct


def draw_summary_banner(image, crop_count, weed_count, coverage_pct):
    """Adds a small readable text banner at the top of the image with the
    key numbers, so the annotated image is useful on its own without
    needing to cross-reference the text report."""
    banner_text = f"Crop: {crop_count}  Weed: {weed_count}  Weed coverage: {coverage_pct:.1f}%"
    cv2.rectangle(image, (0, 0), (image.shape[1], 30), (30, 30, 30), thickness=-1)
    cv2.putText(image, banner_text, (8, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="image file or folder of images")
    parser.add_argument("--weights", default="runs/farmeye/farmeye_train/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--out-dir", default="farmeye_results")
    parser.add_argument("--report", default="farmeye_report.txt")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    model = YOLO(args.weights)
    results = model.predict(source=args.source, conf=args.conf, save=False, verbose=False)

    report_lines = []
    grand_total = Counter()
    coverage_values = []

    for result in results:
        image_name = Path(result.path).name
        class_counts, coverage_pct = summarize_image(result)

        crop_count = class_counts.get("crop", 0)
        weed_count = class_counts.get("weed", 0)
        grand_total["crop"] += crop_count
        grand_total["weed"] += weed_count
        coverage_values.append(coverage_pct)

        # draw boxes manually rather than relying on Ultralytics' default
        # plot() styling, so we can add our own banner underneath
        annotated = result.orig_img.copy()
        for box in result.boxes:
            cls_name = result.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            color = BOX_COLOR.get(cls_name, (255, 255, 255))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, cls_name, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        annotated = draw_summary_banner(annotated, crop_count, weed_count, coverage_pct)
        cv2.imwrite(str(out_dir / image_name), annotated)

        line = f"{image_name}: crop={crop_count}, weed={weed_count}, weed_coverage={coverage_pct:.1f}%"
        report_lines.append(line)
        print(line)

    avg_coverage = sum(coverage_values) / len(coverage_values) if coverage_values else 0
    summary = (
        "\n=== FarmEye-AI Summary ===\n"
        f"Images processed: {len(results)}\n"
        f"Total crop detections: {grand_total['crop']}\n"
        f"Total weed detections: {grand_total['weed']}\n"
        f"Average weed coverage across images: {avg_coverage:.1f}%\n"
    )
    print(summary)

    report_path = Path(args.report)
    report_path.write_text("\n".join(report_lines) + "\n" + summary)

    print(f"Annotated images saved to: {out_dir}/")
    print(f"Text report saved to: {report_path}")


if __name__ == "__main__":
    main()
