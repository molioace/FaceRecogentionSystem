import cv2
import numpy as np
from skimage import transform as trans


class FaceAligner:
    ARC_FACE_DST = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ], dtype=np.float32)

    def __init__(self, image_size=112):
        self.image_size = image_size

    def align(self, image, five_landmarks):
        if five_landmarks.shape != (5, 2):
            raise ValueError(f"Expected landmarks shape (5,2), got {five_landmarks.shape}")

        tform = trans.SimilarityTransform()
        success = tform.estimate(
            five_landmarks.astype(np.float32),
            self.ARC_FACE_DST
        )

        if not success:
            raise RuntimeError("Face alignment failed.")

        matrix = tform.params[0:2, :]

        aligned_face = cv2.warpAffine(
            image,
            matrix,
            (self.image_size, self.image_size),
            flags=cv2.INTER_LINEAR,
            borderValue=0.0
        )

        return aligned_face