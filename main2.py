from ultralytics import YOLO
import cv2

# Pre-trained YOLOv8 model load karo
model = YOLO("yolov8n.pt")  # small & fast model, pehle isse try kar

# Webcam open karo
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)

    # Frame pe bounding boxes draw karwao
    annotated_frame = results[0].plot()

    cv2.imshow("Object Identifier", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
