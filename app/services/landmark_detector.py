import cv2
import numpy as np
import onnxruntime as ort

from app.config import LANDMARK_MODEL_PATH


class LandmarkDetector:
    def __init__(self):
        self.session = ort.InferenceSession(
            str(LANDMARK_MODEL_PATH),
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, face_crop):
        face = cv2.resize(face_crop, (192, 192))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        face = face.astype(np.float32)
        face = (face - 127.5) / 128.0

        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0)

        return face

    def detect(self, face_crop, crop_bbox):
        """
        Returns 5 ArcFace landmarks in original frame coordinates.
        """

        x1, y1, x2, y2 = crop_bbox
        crop_w = x2 - x1
        crop_h = y2 - y1

        input_tensor = self.preprocess(face_crop)

        output = self.session.run(
            None,
            {self.input_name: input_tensor}
        )[0]

        landmarks_106 = output[0].reshape(106, 2)

        # Convert from 192x192 crop space back to original image space
        landmarks_106[:, 0] = landmarks_106[:, 0] * crop_w / 192 + x1
        landmarks_106[:, 1] = landmarks_106[:, 1] * crop_h / 192 + y1

        # Common 106-point to 5-point mapping
        left_eye = landmarks_106[38]
        right_eye = landmarks_106[88]
        nose = landmarks_106[86]
        left_mouth = landmarks_106[52]
        right_mouth = landmarks_106[61]

        five_landmarks = np.array([
            left_eye,
            right_eye,
            nose,
            left_mouth,
            right_mouth
        ], dtype=np.float32)

        return five_landmarks