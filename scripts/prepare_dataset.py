"""
Dataset Hazırlama Ana Scripti
=============================
Çalıştırma:
  python scripts/prepare_dataset.py
  python scripts/prepare_dataset.py --seed 42 --stats

Bu script:
  1. JSON → YOLO segmentation formatına çevirir
  2. Hasta bazlı train/val/test split uygular
  3. Görüntü + etiket dosyalarını dataset/ klasörüne kopyalar
  4. configs/dataset.yaml içindeki path'i günceller
  5. İstatistik raporu basar
"""

import argparse
import json
import logging
import shutil
import sys
from collections import Counter
from pathlib import Path

# Proje kökünü Python path'e ekle (hem lokal hem Kaggle uyumlu)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.paths import PATHS
from src.data.convert import CLASS_MAP, labelme_to_yolo
from src.data.split import build_split, get_split_summary

# ──────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────
# Yardımcı
# ──────────────────────────────────────────────────
def _update_dataset_yaml(dataset_path: Path, yaml_path: Path) -> None:
    """dataset.yaml içindeki 'path' alanını çalışma ortamına göre günceller."""
    text = yaml_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    new_lines = []
    for line in lines:
        if line.strip().startswith("path:"):
            new_lines.append(f"path: {dataset_path.as_posix()}")
        else:
            new_lines.append(line)
    yaml_path.write_text("\n".join(new_lines), encoding="utf-8")
    logger.info("dataset.yaml güncellendi → path: %s", dataset_path)


def _print_stats(splits: dict, label_dirs: dict[str, Path]) -> None:
    """Her split için görüntü ve sınıf dağılım istatistiği basar."""
    CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}
    total_classes = {name: Counter() for name in splits}

    print("\n" + "=" * 65)
    print(f"{'Split':<7} {'Görüntü':>7} {'Sperm':>7}  " + "  ".join(f"{CLASS_NAMES[i]:>5}" for i in range(5)))
    print("-" * 65)

    grand_total_img = 0
    grand_total_sperm = 0

    for split_name, json_paths in splits.items():
        lbl_dir = label_dirs[split_name]
        n_img = len(json_paths)
        grand_total_img += n_img
        class_counts = Counter()

        for jp in json_paths:
            txt = lbl_dir / f"{jp.stem}.txt"
            if txt.exists():
                for line in txt.read_text().splitlines():
                    if line.strip():
                        cid = int(line.split()[0])
                        class_counts[cid] += 1

        n_sperm = sum(class_counts.values())
        grand_total_sperm += n_sperm
        total_classes[split_name] = class_counts

        counts_str = "  ".join(f"{class_counts.get(i, 0):>5}" for i in range(5))
        print(f"{split_name:<7} {n_img:>7} {n_sperm:>7}  {counts_str}")

    print("=" * 65)
    print(f"{'TOPLAM':<7} {grand_total_img:>7} {grand_total_sperm:>7}\n")


# ──────────────────────────────────────────────────
# Ana fonksiyon
# ──────────────────────────────────────────────────
def prepare(seed: int = 42, show_stats: bool = True, clean: bool = False) -> None:

    raw_json    = PATHS["raw_json"]
    raw_images  = PATHS["raw_images"]
    dataset_dir = PATHS["dataset"]
    yaml_path   = PATHS["configs"] / "dataset.yaml"

    # Mevcut dataset'i temizle
    if clean and dataset_dir.exists():
        logger.warning("Mevcut dataset/ klasörü temizleniyor…")
        shutil.rmtree(dataset_dir)

    # Klasörleri oluştur
    label_dirs = {
        "train": PATHS["labels_train"],
        "val":   PATHS["labels_val"],
        "test":  PATHS["labels_test"],
    }
    image_dirs = {
        "train": PATHS["images_train"],
        "val":   PATHS["images_val"],
        "test":  PATHS["images_test"],
    }
    for d in list(label_dirs.values()) + list(image_dirs.values()):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Hasta bazlı split
    logger.info("Split hesaplanıyor… (seed=%d)", seed)
    splits = build_split(raw_json, seed=seed)
    print(get_split_summary(splits))

    # 2. JSON → YOLO dönüşümü + dosya kopyalama
    total_stats = Counter()

    for split_name, json_paths in splits.items():
        logger.info("▶ %s (%d görüntü) işleniyor…", split_name, len(json_paths))

        for json_path in json_paths:
            stem = json_path.stem

            # Etiket dönüştür
            txt_out = label_dirs[split_name] / f"{stem}.txt"
            conv_stats = labelme_to_yolo(
                json_path=json_path,
                output_txt_path=txt_out,
            )
            total_stats["converted"] += conv_stats["converted"]
            total_stats["skipped"]   += conv_stats["skipped"]

            # Görüntüyü kopyala
            img_src = raw_images / f"{stem}.jpg"
            if img_src.exists():
                shutil.copy2(img_src, image_dirs[split_name] / img_src.name)
                total_stats["images_copied"] += 1
            else:
                logger.warning("Görüntü bulunamadı: %s", img_src.name)
                total_stats["images_missing"] += 1

    logger.info(
        "Dönüşüm tamamlandı: %d anotasyon, %d atlandı, %d görüntü kopyalandı, %d eksik",
        total_stats["converted"],
        total_stats["skipped"],
        total_stats["images_copied"],
        total_stats["images_missing"],
    )

    # 3. dataset.yaml güncelle
    _update_dataset_yaml(dataset_dir, yaml_path)

    # 4. İstatistikler
    if show_stats:
        _print_stats(splits, label_dirs)

    logger.info("✅ Dataset hazır → %s", dataset_dir)


# ──────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset hazırlama scripti")
    parser.add_argument("--seed",  type=int,  default=42,    help="Rastgele tohum (default: 42)")
    parser.add_argument("--stats", action="store_true",      help="Sınıf dağılımı istatistikleri göster")
    parser.add_argument("--clean", action="store_true",      help="Mevcut dataset/ klasörünü temizle ve yeniden oluştur")
    args = parser.parse_args()

    prepare(seed=args.seed, show_stats=args.stats, clean=args.clean)
