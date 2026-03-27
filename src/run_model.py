from ultralytics import YOLO
import cv2

MODEL_PATH = r"D:\Dev\Universität\turbine_detection\runs\detect\train5\weights\best.pt"
CAMERA_INDEX = 3
CONFIDENCE = 0.005


def main():
    print("Lade Modell...")
    model = YOLO(MODEL_PATH)

    print("Öffne Kamera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"Fehler: Kamera {CAMERA_INDEX} konnte nicht geöffnet werden.")
        return

    print("Starte Live-Erkennung. Drücke q zum Beenden.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Fehler: Kein Kameraframe gelesen.")
            break

        results = model.predict(frame, conf=CONFIDENCE, verbose=False)
        result = results[0]
        annotated = result.plot()

        num_boxes = 0 if result.boxes is None else len(result.boxes)

        if result.boxes is not None and len(result.boxes) > 0:
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                print(f"Erkannt: {names[cls_id]} ({conf:.2f})")

        cv2.putText(
            annotated,
            f"Detections: {num_boxes}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        cv2.imshow("YOLO11 Live Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()