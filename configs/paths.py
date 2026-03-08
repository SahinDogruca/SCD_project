"""
Ortam tespiti: Kaggle ve lokal için otomatik path yönetimi.
Kaggle'da /kaggle/input/<dataset-name>/ yapısına göre ayarlanır.
"""

import os
from pathlib import Path


def _is_kaggle() -> bool:
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None or Path("/kaggle").exists()


def get_paths() -> dict:
    """Çalışma ortamına göre doğru path'leri döndürür."""

    if _is_kaggle():
        # Kaggle'da dataset input klasörü
        # Dataset adını kendi Kaggle dataset adınızla değiştirin
        KAGGLE_DATASET = os.environ.get("KAGGLE_DATASET_NAME", "oksidatif-stress-sperm")
        raw_root = Path(f"/kaggle/input/{KAGGLE_DATASET}")
        project_root = Path("/kaggle/working")
    else:
        # Lokal - bu dosyanın iki üst dizini = proje kökü
        project_root = Path(__file__).resolve().parent.parent
        raw_root = project_root  # Ham veri proje kökünde

    return {
        # Ham veri
        "raw_images":    raw_root / "images",
        "raw_json":      raw_root / "json_files",
        "raw_masks":     raw_root / "masks",

        # İşlenmiş dataset (YOLO formatı)
        "dataset":       project_root / "dataset",
        "images_train":  project_root / "dataset" / "images" / "train",
        "images_val":    project_root / "dataset" / "images" / "val",
        "images_test":   project_root / "dataset" / "images" / "test",
        "labels_train":  project_root / "dataset" / "labels" / "train",
        "labels_val":    project_root / "dataset" / "labels" / "val",
        "labels_test":   project_root / "dataset" / "labels" / "test",

        # Çıktılar
        "runs":          project_root / "runs",
        "configs":       project_root / "configs",
        "project_root":  project_root,
    }


# Tek import ile kullanım: from configs.paths import PATHS
PATHS = get_paths()
