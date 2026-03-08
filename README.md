# Sperm SCD Segmentation

Sperm Chromatin Dispersion (SCD) testinden elde edilen mikroskopi görüntülerinde hale genişliğine göre sperm segmentasyonu.

## Sınıflar
| ID | Etiket | Açıklama |
|----|--------|---------|
| 0 | NH | No Halo — hale yok |
| 1 | SH | Small Halo — küçük hale |
| 2 | MH | Medium Halo — orta hale |
| 3 | BH | Big Halo — büyük hale |
| 4 | DEG | Degenerated — dejenere sperm |

## Proje Yapısı
```
OksidatifStress/
├── configs/
│   ├── paths.py          # Lokal/Kaggle ortam tespiti
│   └── dataset.yaml      # YOLO dataset konfigürasyonu
├── src/
│   └── data/
│       ├── convert.py    # LabelMe JSON → YOLO formatı
│       └── split.py      # Hasta bazlı train/val/test split
├── scripts/
│   ├── prepare_dataset.py  # Veriyi hazırla (önce bunu çalıştır)
│   └── train_yolo.py       # YOLOv12 eğitimi
├── dataset/              # Hazır veri (prepare_dataset.py oluşturur)
│   ├── images/{train,val,test}/
│   └── labels/{train,val,test}/
├── runs/                 # Eğitim çıktıları
└── requirements.txt
```

## Kurulum
```bash
pip install -r requirements.txt
```

## Kullanım

### 1. Dataset Hazırla
```bash
python scripts/prepare_dataset.py --stats --clean
```

### 2. YOLOv12 Eğit
```bash
# Hızlı test (lokal)
python scripts/train_yolo.py --model yolov12s-seg.pt --epochs 20 --device cpu

# Tam eğitim (GPU)
python scripts/train_yolo.py --model yolov12m-seg.pt --epochs 100
```

## Kaggle

1. Veri setini Kaggle'a yükleyin
2. `KAGGLE_DATASET_NAME` ortam değişkenini dataset adınızla ayarlayın
3. Scriptleri aynen çalıştırın — path'ler otomatik ayarlanır

## Split Stratejisi
- **Train**: Hidayet, M, Sefa, i, O + U'nun %50'si (~370 görüntü)
- **Val**: U'nun %50'si (~13 görüntü)  
- **Test**: Omer, Ozgur, Uzeyir — tamamen görülmemiş hasta grubu (6 görüntü)
