from fastapi import APIRouter, UploadFile, File
import cv2
import numpy as np

from app.services.face_pipeline import FaceRecognitionPipeline

router = APIRouter()
pipeline = FaceRecognitionPipeline()


def read_image(file):
    contents = file.file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


@router.post("/recognize")
def recognize(file: UploadFile = File(...)):
    try:
        frame = read_image(file)
        return pipeline.recognize(frame)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e)
        }


@router.post("/register")
def register(user_id: str, file: UploadFile = File(...)):
    try:
        frame = read_image(file)
        return pipeline.register(frame, user_id)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e)
        }


@router.get("/health")
def health():
    return {"status": "ok"}