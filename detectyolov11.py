from ultralytics import YOLO
import cv2

model = YOLO("C:/Users/syahla/Downloads/best.pt")

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform object detection
    results = model(frame, show=True)

    frame_with_detections = results[0].plot()
    cv2.imshow("Detections", frame_with_detections)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break