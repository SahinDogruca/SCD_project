"""
YOLO + SAM2 Inference Pipeline
================================
YOLO → bounding box + sınıf etiketi
SAM2 → o box'ı prompt olarak kullanarak piksel hassasiyetinde maske üret

Kurulum:
  pip install ultralytics  # SAM2 dahil

Kullanım:
  # Tek görüntü predict
  python scripts/sam_inference.py --source image.jpg --yolo-weights best.pt

  # Klasör predict
  python scripts/sam_inference.py --source dataset/images/test/ --yolo-weights best.pt

  # Test seti mAP değerlendirme
  python scripts/sam_inference.py --evaluate --yolo-weights best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.paths import PATHS

# Sınıf eşleşmeleri
CLASS_NAMES = {0: "NH", 1: "SH", 2: "MH", 3: "BH", 4: "DEG"}
CLASS_COLORS = {
    0: (100, 200, 100),   # NH  — yeşil
    1: (100, 150, 255),   # SH  — mavi
    2: (255, 180, 50),    # MH  — turuncu
    3: (80,  200, 230),   # BH  — cyan
    4: (220,  60, 100),   # DEG — kırmızı
}


# ──────────────────────────────────────────────────
# Model yükle
# ──────────────────────────────────────────────────
def load_models(yolo_weights: str, sam_model: str = "sam2.1_l.pt"):
    from ultralytics import SAM, YOLO
    yolo = YOLO(yolo_weights)
    sam  = SAM(sam_model)
    return yolo, sam


# ──────────────────────────────────────────────────
# Tek görüntü üzerinde YOLO → SAM pipeline
# ──────────────────────────────────────────────────
def predict_image(
    image_path: Path,
    yolo_model,
    sam_model,
    conf: float = 0.25,
    iou:  float = 0.45,
) -> dict:
    """
    YOLO ile detect et, SAM ile maske üret.

    Returns:
        {
          'boxes'   : [[x1,y1,x2,y2], ...],
          'classes' : [0, 1, ...],
          'scores'  : [0.95, ...],
          'masks'   : [H×W bool array, ...],
        }
    """
    # 1. YOLO ile tespit (sadece bbox + class kullanıyoruz)
    yolo_results = yolo_model.predict(
        str(image_path),
        conf=conf,
        iou=iou,
        verbose=False,
    )[0]

    if yolo_results.boxes is None or len(yolo_results.boxes) == 0:
        return {"boxes": [], "classes": [], "scores": [], "masks": []}

    boxes   = yolo_results.boxes.xyxy.cpu().numpy()    # [N, 4]
    classes = yolo_results.boxes.cls.cpu().numpy().astype(int)
    scores  = yolo_results.boxes.conf.cpu().numpy()

    # 2. SAM'a box prompt olarak ver → hassas maskeler
    sam_results = sam_model.predict(
        str(image_path),
        bboxes=boxes,
        verbose=False,
    )[0]

    # SAM maskeleri binary array olarak al
    if sam_results.masks is not None:
        masks = sam_results.masks.data.cpu().numpy().astype(bool)  # [N, H, W]
    else:
        # SAM sonuç vermediyse YOLO'nun kendi maskelerini kullan
        masks = []
        if yolo_results.masks is not None:
            masks = yolo_results.masks.data.cpu().numpy().astype(bool)

    return {
        "boxes":   boxes.tolist(),
        "classes": classes.tolist(),
        "scores":  scores.tolist(),
        "masks":   masks,
    }


# ──────────────────────────────────────────────────
# Görüntü vizualizasyonu
# ──────────────────────────────────────────────────
def draw_predictions(image_path: Path, predictions: dict, alpha: float = 0.45) -> np.ndarray:
    """Tahminleri görüntü üzerine çizer."""
    img = cv2.imread(str(image_path))
    overlay = img.copy()

    boxes   = predictions["boxes"]
    classes = predictions["classes"]
    scores  = predictions["scores"]
    masks   = predictions["masks"]

    for i, (box, cls_id, score) in enumerate(zip(boxes, classes, scores)):
        color = CLASS_COLORS.get(cls_id, (200, 200, 200))
        label = f"{CLASS_NAMES.get(cls_id, '?')} {score:.2f}"

        # Maske
        if i < len(masks):
            mask = masks[i]
            if mask.shape[:2] != img.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            overlay[mask] = color

        # Bounding box
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

        # Etiket
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(img, label, (x1 + 1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Maske katmanını görüntüyle birleştir
    result = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return result


# ──────────────────────────────────────────────────
# Test Seti mAP Değerlendirme
# ──────────────────────────────────────────────────
def evaluate_test_set(yolo_model, sam_model, conf: float = 0.25) -> dict:
    """
    Test setindeki tüm görüntüler için YOLO+SAM ile predict et,
    YOLO ground truth etiketleriyle karşılaştır ve sınıf başına istatistik ver.

    Not: Tam mAP@50-95 hesabı için pycocotools kullanılır.
    Bu fonksiyon daha sade bir precision/recall özeti verir.
    """
    from collections import defaultdict

    test_img_dir = PATHS["images_test"]
    test_lbl_dir = PATHS["labels_test"]

    image_files = sorted(test_img_dir.glob("*.jpg"))
    print(f"\n📊 Test seti değerlendirme: {len(image_files)} görüntü\n")

    per_class_tp = defaultdict(int)
    per_class_fp = defaultdict(int)
    per_class_fn = defaultdict(int)
    iou_threshold = 0.5

    for img_path in image_files:
        lbl_path = test_lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        # Ground truth
        gt_boxes, gt_classes = _load_yolo_labels(lbl_path, img_path)

        # Prediction
        preds = predict_image(img_path, yolo_model, sam_model, conf=conf)
        pred_boxes   = np.array(preds["boxes"])   if preds["boxes"]   else np.zeros((0, 4))
        pred_classes = np.array(preds["classes"]) if preds["classes"] else np.array([])

        # Basit IoU eşleştirme
        matched_gt = set()
        for pi, (pb, pc) in enumerate(zip(pred_boxes, pred_classes)):
            best_iou, best_gi = 0.0, -1
            for gi, (gb, gc) in enumerate(zip(gt_boxes, gt_classes)):
                if gc != pc or gi in matched_gt:
                    continue
                iou = _box_iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= iou_threshold and best_gi >= 0:
                per_class_tp[int(pc)] += 1
                matched_gt.add(best_gi)
            else:
                per_class_fp[int(pc)] += 1

        for gi, gc in enumerate(gt_classes):
            if gi not in matched_gt:
                per_class_fn[int(gc)] += 1

    # Özet
    print(f"{'Sınıf':<8} {'TP':>6} {'FP':>6} {'FN':>6} {'Precision':>10} {'Recall':>8}")
    print("-" * 50)
    all_tp = all_fp = all_fn = 0
    for cid, cname in CLASS_NAMES.items():
        tp = per_class_tp[cid]
        fp = per_class_fp[cid]
        fn = per_class_fn[cid]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"{cname:<8} {tp:>6} {fp:>6} {fn:>6} {prec:>10.3f} {rec:>8.3f}")
        all_tp += tp; all_fp += fp; all_fn += fn

    prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    rec  = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    print("-" * 50)
    print(f"{'ALL':<8} {all_tp:>6} {all_fp:>6} {all_fn:>6} {prec:>10.3f} {rec:>8.3f}")
    return {"per_class_tp": dict(per_class_tp), "precision": prec, "recall": rec}


def _load_yolo_labels(lbl_path: Path, img_path: Path):
    """YOLO .txt etiketini yükle, absolute bbox koordinatlarına çevir."""
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    boxes, classes = [], []
    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        coords = list(map(float, parts[1:]))
        # Polygon → bbox (min/max)
        xs = coords[0::2]
        ys = coords[1::2]
        x1, y1 = min(xs) * w, min(ys) * h
        x2, y2 = max(xs) * w, max(ys) * h
        boxes.append([x1, y1, x2, y2])
        classes.append(cls)
    return np.array(boxes), np.array(classes)


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """İki bbox arasında IoU hesapla."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ──────────────────────────────────────────────────
# Klasör tahminleri → çıktı kaydet
# ──────────────────────────────────────────────────
def predict_folder(
    source_dir: Path,
    yolo_model,
    sam_model,
    output_dir: Path,
    conf: float = 0.25,
    save_json: bool = True,
    save_images: bool = True,
) -> None:
    """Bir klasördeki tüm görüntüler için predict et ve kaydet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images_out = output_dir / "images"
    json_out   = output_dir / "json"
    if save_images: images_out.mkdir(exist_ok=True)
    if save_json:   json_out.mkdir(exist_ok=True)

    image_files = sorted(source_dir.glob("*.jpg")) + sorted(source_dir.glob("*.png"))
    print(f"🔍 {len(image_files)} görüntü işleniyor → {output_dir}")

    for i, img_path in enumerate(image_files):
        preds = predict_image(img_path, yolo_model, sam_model, conf=conf)

        if save_images:
            vis = draw_predictions(img_path, preds)
            cv2.imwrite(str(images_out / img_path.name), vis)

        if save_json:
            # Maskeleri JSON'a yazma (çok büyük), sadece box+class+score
            json_data = {
                "file": img_path.name,
                "predictions": [
                    {
                        "class_id":   int(c),
                        "class_name": CLASS_NAMES.get(int(c), "?"),
                        "score":      float(s),
                        "box_xyxy":   [float(v) for v in b],
                    }
                    for c, s, b in zip(
                        preds["classes"], preds["scores"], preds["boxes"]
                    )
                ],
            }
            (json_out / f"{img_path.stem}.json").write_text(
                json.dumps(json_data, indent=2)
            )

        if (i + 1) % 10 == 0 or (i + 1) == len(image_files):
            print(f"  [{i+1}/{len(image_files)}] ✓")

    print(f"\n✅ Tamamlandı → {output_dir}")


# ──────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO + SAM2 Inference Pipeline")
    parser.add_argument("--yolo-weights", required=True,
                        help="YOLO model ağırlık dosyası (.pt)")
    parser.add_argument("--sam-model",    default="sam2.1_b.pt",
                        help="SAM model adı (default: sam2.1_b.pt)")
    parser.add_argument("--source",       default=None,
                        help="Görüntü dosyası veya klasörü")
    parser.add_argument("--output",       default="runs/sam_inference",
                        help="Çıktı klasörü (default: runs/sam_inference)")
    parser.add_argument("--conf",         type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--evaluate",     action="store_true",
                        help="Test seti üzerinde değerlendir")
    args = parser.parse_args()

    print("📦 Modeller yükleniyor…")
    yolo, sam = load_models(args.yolo_weights, args.sam_model)

    if args.evaluate:
        evaluate_test_set(yolo, sam, conf=args.conf)

    if args.source:
        source = Path(args.source)
        output = PATHS["project_root"] / args.output
        if source.is_file():
            preds = predict_image(source, yolo, sam, conf=args.conf)
            vis   = draw_predictions(source, preds)
            out   = output / source.name
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), vis)
            print(f"✅ Kaydedildi → {out}")
            print(f"   Tespit: {len(preds['boxes'])} sperm")
            for c, s in zip(preds["classes"], preds["scores"]):
                print(f"   {CLASS_NAMES.get(c,'?')}: {s:.3f}")
        elif source.is_dir():
            predict_folder(source, yolo, sam, output, conf=args.conf)
        else:
            print(f"❌ Hata: '{source}' bulunamadı.")
