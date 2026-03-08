"""
Hasta bazlı Train / Val / Test split.

Strateji:
  - Train : Hidayet, M, Sefa, i, O  + U'nun ilk yarısı
  - Val   : U'nun ikinci yarısı
  - Test  : Ozgur, Uzeyir, Omer

Neden U ikiye bölünüyor?
  U en büyük ikinci hasta (26 görüntü). Tümünü val'e koymak yerine
  yarısını train'e alarak val'i hem dengeli hem de yeterli büyüklükte tutuyoruz.
  Aynı hasta içi bölünme, hasta arası data leakage oluşturmaz.
"""

import re
import logging
import random
import shutil
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────
# Hasta → Split eşleşmesi
# ──────────────────────────────────────────────────
PATIENT_SPLIT: dict[str, str] = {
    "Hidayet": "train",
    "M":       "train",
    "Sefa":    "train",
    "i":       "train",
    "O":       "train",
    # U aşağıda dinamik olarak bölünür
    "Omer":    "test",
    "Ozgur":   "test",
    "Uzeyir":  "test",
}

# U için train'e gidecek oran
U_TRAIN_RATIO = 0.5


def _extract_patient(filename: str) -> str:
    """Dosya adından hasta adını çıkarır. Ör: 'M100.json' → 'M'"""
    m = re.match(r"^([A-Za-z]+)", filename)
    return m.group(1) if m else "Unknown"


def build_split(
    json_dir: Path,
    seed: int = 42,
) -> dict[str, list[Path]]:
    """
    JSON dosya listesini hasta bazlı train/val/test'e böler.

    Returns:
        {"train": [...], "val": [...], "test": [...]}
    """
    random.seed(seed)

    # Hasta → dosyalar eşleşmesi
    patient_files: dict[str, list[Path]] = defaultdict(list)
    for json_path in sorted(json_dir.glob("*.json")):
        patient = _extract_patient(json_path.stem)
        patient_files[patient].append(json_path)

    splits: dict[str, list[Path]] = {"train": [], "val": [], "test": []}

    for patient, files in patient_files.items():
        files = sorted(files)  # Tekrarlanabilir sıra

        if patient in PATIENT_SPLIT:
            target = PATIENT_SPLIT[patient]
            splits[target].extend(files)

        elif patient == "U":
            # U'yu deterministik olarak ikiye böl (shuffle + split)
            random.shuffle(files)
            cut = int(len(files) * U_TRAIN_RATIO)
            splits["train"].extend(files[:cut])
            splits["val"].extend(files[cut:])

        else:
            logger.warning("Tanınmayan hasta '%s' — train'e eklendi.", patient)
            splits["train"].extend(files)

    # Özet log
    for split_name, file_list in splits.items():
        logger.info("%-5s : %3d görüntü", split_name, len(file_list))

    return splits


def copy_split_files(
    splits: dict[str, list[Path]],
    json_dir: Path,
    images_dir: Path,
    out_images: dict[str, Path],
    out_labels: dict[str, Path],
    label_suffix: str = ".txt",
) -> dict[str, dict]:
    """
    Split edilen dosyaları hedef klasörlere kopyalar.
    Görüntüler → dataset/images/{split}/
    Etiketler  → dataset/labels/{split}/  (zaten dönüştürülmüş .txt)

    Args:
        splits: build_split() çıktısı.
        json_dir: Kaynak JSON klasörü (sadece referans için).
        images_dir: Kaynak görüntü klasörü.
        out_images: {"train": Path, "val": Path, "test": Path}
        out_labels: {"train": Path, "val": Path, "test": Path}
        label_suffix: Etiket dosyası uzantısı (varsayılan .txt).

    Returns:
        Her split için {copied, missing} istatistikleri.
    """
    stats = {}

    for split_name, json_paths in splits.items():
        img_out = out_images[split_name]
        lbl_out = out_labels[split_name]
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        copied, missing = 0, 0

        for json_path in json_paths:
            stem = json_path.stem

            # Görüntü dosyası
            img_src = images_dir / f"{stem}.jpg"
            if img_src.exists():
                shutil.copy2(img_src, img_out / img_src.name)
                copied += 1
            else:
                logger.warning("Görüntü bulunamadı: %s", img_src)
                missing += 1

            # Etiket dosyası (daha önce dönüştürülmüş .txt)
            lbl_src = json_path.parent.parent / "dataset" / "labels_tmp" / f"{stem}{label_suffix}"
            # Not: prepare_dataset.py bu geçici yolu yönetir

        stats[split_name] = {"image_count": len(json_paths), "copied": copied, "missing": missing}
        logger.info(
            "%-5s → %d dosya kopyalandı, %d eksik",
            split_name, copied, missing,
        )

    return stats


def get_split_summary(splits: dict[str, list[Path]]) -> str:
    """Okunabilir split özeti döndürür."""
    lines = ["Split Özeti:", "-" * 40]
    total = sum(len(v) for v in splits.values())
    for name, files in splits.items():
        patients = sorted({_extract_patient(f.stem) for f in files})
        pct = len(files) / total * 100 if total else 0
        lines.append(f"  {name:5s}: {len(files):3d} görüntü ({pct:4.1f}%) — {', '.join(patients)}")
    lines.append(f"  {'TOTAL':5s}: {total:3d}")
    return "\n".join(lines)
