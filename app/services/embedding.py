import cv2
import numpy as np
import onnxruntime as ort

from app.config import ARCFACE_MODEL_PATH


class FaceEmbedder:
    def __init__(self):
        self.session = ort.InferenceSession(
            str(ARCFACE_MODEL_PATH),
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, aligned_face):
        face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)

        face = face.astype(np.float32)
        face = (face - 127.5) / 127.5

        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0)

        return face

    def get_embedding(self, aligned_face):
        input_tensor = self.preprocess(aligned_face)

        embedding = self.session.run(
            None,
            {self.input_name: input_tensor}
        )[0]

        embedding = embedding[0].astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        return embedding