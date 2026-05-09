from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"
INDEXES_DIR = BASE_DIR / "indexes"

YOLO_MODEL_PATH = MODELS_DIR / "yolov8s-face-lindevs.onnx"
ARCFACE_MODEL_PATH = MODELS_DIR / "w600k_r50.onnx"
LANDMARK_MODEL_PATH = MODELS_DIR / "2d106det.onnx"
FAISS_INDEX_PATH = INDEXES_DIR / "faces.index"
