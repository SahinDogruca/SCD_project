"""
LabelMe JSON → YOLO Instance Segmentation formatına dönüşüm.

YOLO segmentation formatı (her satır bir obje):
  class_id  x1 y1 x2 y2 x3 y3 ...  (normalize edilmiş 0-1 arası)
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Sınıf → ID eşleşmesi (sabit tutun, modeller buna göre eğitilir)
CLASS_MAP: dict[str, int] = {
    "NH":  0,
    "SH":  1,
    "MH":  2,
    "BH":  3,
    "DEG": 4,
}


def _normalize_points(points: list[list[float]], width: int, height: int) -> list[float]:
    """Polygon noktalarını görüntü boyutuna göre [0,1] aralığına normalize eder."""
    normalized = []
    for x, y in points:
        normalized.append(round(x / width, 6))
        normalized.append(round(y / height, 6))
    return normalized


def _clamp(values: list[float]) -> list[float]:
    """Görüntü sınırı dışına taşan koordinatları [0,1] içinde tutar."""
    return [max(0.0, min(1.0, v)) for v in values]


def labelme_to_yolo(
    json_path: Path,
    output_txt_path: Path,
    class_map: dict[str, int] = CLASS_MAP,
    skip_unknown: bool = True,
) -> dict:
    """
    Tek bir LabelMe JSON dosyasını YOLO segmentation .txt formatına çevirir.

    Args:
        json_path: Giriş LabelMe JSON dosyası.
        output_txt_path: Çıkış YOLO .txt dosyası.
        class_map: Etiket → sınıf ID eşleşmesi.
        skip_unknown: Bilinmeyen etiketleri atla (True) veya hata ver (False).

    Returns:
        {total, converted, skipped} istatistik dict'i.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    width: int = data["imageWidth"]
    height: int = data["imageHeight"]
    shapes: list[dict] = data.get("shapes", [])

    stats = {"total": len(shapes), "converted": 0, "skipped": 0}
    lines: list[str] = []

    for shape in shapes:
        label = shape["label"]
        shape_type = shape.get("shape_type", "polygon")

        # Sadece polygon desteklenir
        if shape_type != "polygon":
            logger.debug("Polygon olmayan shape atlandı: %s (%s)", label, shape_type)
            stats["skipped"] += 1
            continue

        if label not in class_map:
            if skip_unknown:
                logger.warning("Bilinmeyen etiket atlandı: '%s' — %s", label, json_path.name)
                stats["skipped"] += 1
                continue
            else:
                raise ValueError(f"Bilinmeyen etiket: '{label}'. class_map: {list(class_map)}")

        points = shape["points"]
        if len(points) < 3:
            logger.warning("Yetersiz nokta (%d) — atlandı: %s", len(points), json_path.name)
            stats["skipped"] += 1
            continue

        class_id = class_map[label]
        coords = _clamp(_normalize_points(points, width, height))
        coords_str = " ".join(map(str, coords))
        lines.append(f"{class_id} {coords_str}")
        stats["converted"] += 1

    # Boş dosya da oluştur (görüntüde hiç anotasyon yoksa YOLO bunu bekler)
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return stats


def get_image_path_from_json(json_path: Path, images_dir: Path) -> Optional[Path]:
    """JSON'daki imagePath bilgisine göre orijinal görüntü dosyasını bulur."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    image_name = Path(data.get("imagePath", "")).name
    img_path = images_dir / image_name
    return img_path if img_path.exists() else None
