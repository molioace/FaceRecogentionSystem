from pathlib import Path
import urllib.request

MODELS = {
    "detector.onnx": "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8s-face-lindevs.onnx",
    "2d106det.onnx": "https://huggingface.co/menglaoda/_insightface/resolve/main/2d106det.onnx",
    "arcface.onnx": "https://huggingface.co/FoivosPar/Arc2Face/resolve/da2f1e9aa3954dad093213acfc9ae75a68da6ffd/arcface.onnx",
}

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

for filename, url in MODELS.items():
    save_path = models_dir / filename

    if save_path.exists():
        print(f"{filename} already exists.")
        continue

    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, save_path)

print("Done.")