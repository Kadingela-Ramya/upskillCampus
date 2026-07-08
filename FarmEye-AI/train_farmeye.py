"""
train_farmeye.py  (FarmEye-AI)

Trains the FarmEye-AI weed detector. It's a YOLOv8 model underneath -
there's no point reinventing an object detector from scratch when
Ultralytics' implementation is solid and well-tested. What makes this
"FarmEye-AI" is the data it's trained on, the two classes it cares about
(crop / weed), and everything downstream (coverage reporting etc.) built
around it.

Run it like:
    python train_farmeye.py
    python train_farmeye.py --epochs 50 --model yolov8s.pt
"""

import argparse

import torch
from ultralytics import YOLO


def pick_device():
    if torch.cuda.is_available():
        return 0
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--model", default="yolov8n.pt",
                         help="starting weights - 'n' is fastest, 's'/'m' are more accurate but slower")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--run-name", default="farmeye_train")
    args = parser.parse_args()

    device = pick_device()
    print(f"Training on: {'GPU' if device == 0 else 'CPU'}")

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        patience=20,             # stop early if val loss plateaus for 20 epochs
        project="runs/farmeye",
        name=args.run_name,
        exist_ok=True,
        plots=True,
    )

    print("\nDone training.")
    print(f"Best weights: runs/farmeye/{args.run_name}/weights/best.pt")
    print("Check runs/farmeye/<run_name>/results.png for the loss/mAP curves.")


if __name__ == "__main__":
    main()
