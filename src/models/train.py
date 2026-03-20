"""
train.py — YOLOv8 treningspipeline for KamaliAI
Bruk: python3 src/models/train.py --config configs/config.yaml
"""

import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def train(config_path, resume=None):
    cfg = load_config(config_path)

    dataset_yaml = Path(cfg["dataset"]["root"]) / "dataset.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Fant ikke {dataset_yaml}. Kjør preprocess.py først."
        )

    m   = cfg["model"]
    t   = cfg["train"]
    aug = cfg.get("augmentation", {})

    if resume:
        print(f"▶ Fortsetter fra: {resume}")
        model = YOLO(resume)
    else:
        model_name = m["architecture"] + (".pt" if m.get("pretrained", True) else ".yaml")
        print(f"▶ Laster modell:  {model_name}")
        model = YOLO(model_name)

    print(f"▶ Dataset:        {dataset_yaml}")
    print(f"▶ Epochs:         {t['epochs']}")
    print(f"▶ Device:         {t['device']}")
    print()

    results = model.train(
        data=str(dataset_yaml),
        epochs=t["epochs"],
        imgsz=t["imgsz"],
        batch=t["batch"],
        device=t["device"],
        workers=t.get("workers", 2),
        optimizer=t.get("optimizer", "AdamW"),
        lr0=t.get("lr0", 0.001),
        lrf=t.get("lrf", 0.01),
        momentum=t.get("momentum", 0.937),
        weight_decay=t.get("weight_decay", 0.0005),
        patience=t.get("patience", 20),
        save_period=t.get("save_period", 10),
        project=t.get("project", "runs/train"),
        name=t.get("name", "kamali_det"),
        resume=bool(resume),
        mosaic=aug.get("mosaic", 1.0),
        mixup=aug.get("mixup", 0.1),
        hsv_h=aug.get("hsv_h", 0.015),
        hsv_s=aug.get("hsv_s", 0.7),
        hsv_v=aug.get("hsv_v", 0.4),
        flipud=aug.get("flipud", 0.0),
        fliplr=aug.get("fliplr", 0.5),
        degrees=aug.get("degrees", 0.0),
        translate=aug.get("translate", 0.1),
        scale=aug.get("scale", 0.5),
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n Trening ferdig!")
    print(f"   Beste vekter: {best}")
    print(f"   Neste steg:   python3 src/models/evaluate.py --weights {best}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume", default=None, help="Fortsett fra checkpoint")
    args = parser.parse_args()
    train(args.config, args.resume)


if __name__ == "__main__":
    main()
