import cv2
import requests
import tkinter as tk
from tkinter import filedialog

API_BASE = "http://127.0.0.1:5455/recognition"


def send_image_to_register(image_path: str, user_id: str):
    with open(image_path, "rb") as f:
        files = {"file": ("face.jpg", f, "image/jpeg")}
        response = requests.post(
            f"{API_BASE}/register",
            params={"user_id": user_id},
            files=files,
            timeout=30,
        )

    print("Server response:")
    print(response.json())


def upload_photo(user_id: str):
    root = tk.Tk()
    root.withdraw()

    image_path = filedialog.askopenfilename(
        title="Choose face image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png"),
            ("All files", "*.*"),
        ],
    )

    if not image_path:
        print("No image selected.")
        return

    send_image_to_register(image_path, user_id)


def capture_photo(user_id: str):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera.")
        return

    print("Press SPACE to capture. Press ESC to cancel.")

    saved_path = "captured_register_face.jpg"

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read frame.")
            break

        cv2.putText(
            frame,
            "SPACE = Capture | ESC = Cancel",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Register Face", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            print("Cancelled.")
            break

        if key == 32:
            cv2.imwrite(saved_path, frame)
            print(f"Captured image saved: {saved_path}")
            send_image_to_register(saved_path, user_id)
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    print("Admin Face Registration")
    user_id = input("Enter user ID/name: ").strip()

    if not user_id:
        print("User ID is required.")
        return

    print("\nChoose option:")
    print("1. Capture from camera")
    print("2. Upload image file")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        capture_photo(user_id)
    elif choice == "2":
        upload_photo(user_id)
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()