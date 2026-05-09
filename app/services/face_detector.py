import cv2
import numpy as np
import onnxruntime as ort

from app.config import YOLO_MODEL_PATH


class FaceDetector:
    def __init__(self, conf_threshold=0.5, input_size=640):
        self.conf_threshold = conf_threshold
        self.input_size = input_size

        self.session = ort.InferenceSession(
            str(YOLO_MODEL_PATH),
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, image):
        h, w = image.shape[:2]

        resized = cv2.resize(image, (self.input_size, self.input_size))
        img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img, w, h

    def detect(self, image):
        input_tensor, original_w, original_h = self.preprocess(image)

        output = self.session.run(
            None,
            {self.input_name: input_tensor}
        )[0]

        output = output[0].T  # [8400, 5]

        boxes = []
        scores = []

        for pred in output:
            x_center, y_center, box_w, box_h, confidence = pred

            if confidence < self.conf_threshold:
                continue

            x1 = x_center - box_w / 2
            y1 = y_center - box_h / 2
            x2 = x_center + box_w / 2
            y2 = y_center + box_h / 2

            x1 = int(x1 * original_w / self.input_size)
            y1 = int(y1 * original_h / self.input_size)
            x2 = int(x2 * original_w / self.input_size)
            y2 = int(y2 * original_h / self.input_size)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(original_w, x2)
            y2 = min(original_h, y2)

            boxes.append([x1, y1, x2 - x1, y2 - y1])  # OpenCV NMS uses x,y,w,h
            scores.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            score_threshold=self.conf_threshold,
            nms_threshold=0.4
        )

        detections = []

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                detections.append({
                    "bbox": [x, y, x + w, y + h],
                    "detection_confidence": scores[i]
                })

        return detections

    def crop_face(self, image, bbox, padding=0.2):
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox

        box_w = x2 - x1
        box_h = y2 - y1

        x1 = int(x1 - box_w * padding)
        y1 = int(y1 - box_h * padding)
        x2 = int(x2 + box_w * padding)
        y2 = int(y2 + box_h * padding)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        face = image[y1:y2, x1:x2]

        return face, [x1, y1, x2, y2]