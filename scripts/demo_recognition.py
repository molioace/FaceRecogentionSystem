import time
import cv2
import requests

API_BASE = "http://127.0.0.1:5455/recognition"


def call_recognition_api(frame):
    # Resize before sending to reduce backend load
    small = cv2.resize(frame, (640, 480))

    success, encoded_image = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])

    if not success:
        return []

    files = {
        "file": ("frame.jpg", encoded_image.tobytes(), "image/jpeg")
    }

    try:
        response = requests.post(
            f"{API_BASE}/recognize",
            files=files,
            timeout=15,
        )

        if response.status_code != 200:
            print("API status:", response.status_code)
            print("API text:", response.text[:300])
            return []

        try:
            data = response.json()
            print(data)
        except Exception:
            print("Non-JSON response:", response.text[:300])
            return []

        return data.get("results", [])

    except requests.exceptions.ReadTimeout:
        print("API timeout: backend is too slow or overloaded.")
        return []

    except Exception as e:
        print("API error:", e)
        return []


def draw_results(frame, results):
    for result in results:
        bbox = result.get("bbox")

        if not bbox:
            continue

        matched = result.get("matched", False)
        similarity = result.get("similarity", 0.0)

        if matched:
            identity = result.get("identity", "Unknown")
            color = (0, 255, 0)  # green
        else:
            identity = "Unknown"
            color = (0, 0, 255)  # red

        x1, y1, x2, y2 = map(int, bbox)

        label = f"{identity} | {similarity:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label_y1 = max(0, y1 - 35)

        cv2.rectangle(
            frame,
            (x1, label_y1),
            (x2, y1),
            color,
            -1,
        )

        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

    return frame


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera.")
        return

    print("Real-time recognition started.")
    print("Press Q or ESC to quit.")

    last_results = []
    last_api_call = 0
    api_interval = 0.2

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read camera frame.")
            break

        now = time.time()

        if now - last_api_call >= api_interval:
            last_results = call_recognition_api(frame)
            last_api_call = now

        frame = draw_results(frame, last_results)

        cv2.putText(
            frame,
            "Q/ESC = Quit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Real-Time Face Recognition", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()