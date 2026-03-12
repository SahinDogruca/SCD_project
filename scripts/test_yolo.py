from ultralytics import YOLO
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test():
    model = YOLO()

