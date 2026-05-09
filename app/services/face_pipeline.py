from app.services.face_detector import FaceDetector
from app.services.landmark_detector import LandmarkDetector
from app.services.face_aligner import FaceAligner
from app.services.embedding import FaceEmbedder
from app.services.faiss_service import FaceIndex


class FaceRecognitionPipeline:
    def __init__(self):
        self.detector = FaceDetector()
        self.landmarker = LandmarkDetector()
        self.aligner = FaceAligner()
        self.embedder = FaceEmbedder()
        self.index = FaceIndex()

    def recognize(self, frame):
        detections = self.detector.detect(frame)

        if len(detections) == 0:
            return {
                "status": "no_face_detected",
                "faces_count": 0,
                "results": []
            }

        results = []

        for det in detections:
            bbox = det["bbox"]

            face_crop, padded_bbox = self.detector.crop_face(frame, bbox)

            if face_crop.size == 0:
                continue

            five_landmarks = self.landmarker.detect(face_crop, padded_bbox)

            aligned_face = self.aligner.align(frame, five_landmarks)

            embedding = self.embedder.get_embedding(aligned_face)

            match = self.index.search(embedding)

            results.append({
                "bbox": bbox,
                "detection_confidence": det["detection_confidence"],
                "identity": match["identity"],
                "similarity": match["similarity"],
                "matched": match["matched"]
            })

        return {
            "status": "success",
            "faces_count": len(results),
            "results": results
        }

    def register(self, frame, user_id):
        detections = self.detector.detect(frame)

        if len(detections) == 0:
            return {
                "status": "no_face_detected"
            }

        if len(detections) > 1:
            return {
                "status": "multiple_faces_detected",
                "message": "Registration requires exactly one face."
            }

        det = detections[0]
        bbox = det["bbox"]

        face_crop, padded_bbox = self.detector.crop_face(frame, bbox)

        five_landmarks = self.landmarker.detect(face_crop, padded_bbox)

        aligned_face = self.aligner.align(frame, five_landmarks)

        embedding = self.embedder.get_embedding(aligned_face)

        self.index.add_face(user_id, embedding)

        return {
            "status": "registered",
            "user_id": user_id,
            "detection_confidence": det["detection_confidence"]
        }