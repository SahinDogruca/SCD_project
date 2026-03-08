"""
YOLOv12 Instance Segmentation Eğitim Scripti
=============================================
Hem lokal hem Kaggle'da çalışır.

Lokal:
  python scripts/train_yolo.py

Kaggle (notebook hücresinde):
  %run scripts/train_yolo.py --model yolov12s-seg --epochs 100

Ortam değişkenleri ile de kontrol edilebilir:
  YOLO_MODEL=yolov12m-seg YOLO_EPOCHS=50 python scripts/train_yolo.py
"""

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.paths import PATHS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────
# Varsayılan Hiperparametreler
# ──────────────────────────────────────────────────
DEFAULTS = {
    # Model
    "model":      os.environ.get("YOLO_MODEL", "yolov12s-seg.pt"),
    "epochs":     int(os.environ.get("YOLO_EPOCHS", 100)),
    "imgsz":      int(os.environ.get("YOLO_IMGSZ", 1280)),
    "batch":      int(os.environ.get("YOLO_BATCH", -1)),   # -1 = otomatik
    "device":     os.environ.get("YOLO_DEVICE", "0"),      # "0" GPU, "cpu" CPU

    # Augmentation
    "degrees":    180.0,   # Tam rotasyon — sperm yönü önemsiz
    "flipud":     0.5,
    "fliplr":     0.5,
    "scale":      0.5,
    "mosaic":     1.0,
    "copy_paste": 0.4,     # BH/DEG dengesizliği için kritik

    # Boya rengi varyasyonu
    "hsv_h":      0.01,
    "hsv_s":      0.5,
    "hsv_v":      0.4,

    # Optimizasyon
    "optimizer":  "AdamW",
    "lr0":        0.001,
    "lrf":        0.01,    # lr0 * lrf = final lr
    "warmup_epochs": 3,
    "patience":   25,      # Early stopping

    # Kayıp ağırlıkları — sınıf dengesizliği için cls artırıldı
    "box":        7.5,
    "cls":        3.0,     # Varsayılan 0.5 → 3.0 (5 sınıf + dengesizlik)
    "dfl":        1.5,

    # Kayıt
    "save_period": 10,
    "val":         True,
    "plots":       True,
}


def train(cfg: dict) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics paketi bulunamadı. Kurun: pip install ultralytics")
        sys.exit(1)

    yaml_path = PATHS["configs"] / "dataset.yaml"
    if not yaml_path.exists():
        logger.error(
            "dataset.yaml bulunamadı: %s\n"
            "Önce `python scripts/prepare_dataset.py` çalıştırın.",
            yaml_path,
        )
        sys.exit(1)

    runs_dir = PATHS["runs"]
    runs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Model: %s", cfg["model"])
    logger.info("Dataset: %s", yaml_path)
    logger.info("Epochs: %d | imgsz: %d | batch: %s", cfg["epochs"], cfg["imgsz"], cfg["batch"])

    model = YOLO(cfg["model"])

    results = model.train(
        data=str(yaml_path),
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        device=cfg["device"],
        project=str(runs_dir / "yolo"),
        name="sperm_scd",
        exist_ok=False,

        # Augmentations
        degrees=cfg["degrees"],
        flipud=cfg["flipud"],
        fliplr=cfg["fliplr"],
        scale=cfg["scale"],
        mosaic=cfg["mosaic"],
        copy_paste=cfg["copy_paste"],
        hsv_h=cfg["hsv_h"],
        hsv_s=cfg["hsv_s"],
        hsv_v=cfg["hsv_v"],

        # Optimizasyon
        optimizer=cfg["optimizer"],
        lr0=cfg["lr0"],
        lrf=cfg["lrf"],
        warmup_epochs=cfg["warmup_epochs"],
        patience=cfg["patience"],

        # Kayıp
        box=cfg["box"],
        cls=cfg["cls"],
        dfl=cfg["dfl"],

        # Kayıt
        save_period=cfg["save_period"],
        val=cfg["val"],
        plots=cfg["plots"],
        verbose=True,
    )

    logger.info("✅ Eğitim tamamlandı → %s", results.save_dir)

    # Test seti değerlendirmesi
    logger.info("Test seti değerlendiriliyor…")
    model.val(data=str(yaml_path), split="test")


# ──────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv12 Segmentation Eğitimi")
    parser.add_argument("--model",   default=DEFAULTS["model"],   help="Model adı (ör: yolov12s-seg.pt)")
    parser.add_argument("--epochs",  default=DEFAULTS["epochs"],  type=int)
    parser.add_argument("--imgsz",   default=DEFAULTS["imgsz"],   type=int)
    parser.add_argument("--batch",   default=DEFAULTS["batch"],   type=int, help="-1 = otomatik")
    parser.add_argument("--device",  default=DEFAULTS["device"],  help="'0' GPU, 'cpu' CPU")
    parser.add_argument("--copy-paste", default=DEFAULTS["copy_paste"], type=float, dest="copy_paste")
    args = parser.parse_args()

    cfg = {**DEFAULTS, **vars(args)}
    train(cfg)
