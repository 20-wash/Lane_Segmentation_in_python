from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('face_detection/best.pt')

# Open the webcam (0 for default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get the frames per second (fps) of the webcam
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Loop to continuously grab frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict faces in the current frame
    results = model.predict(frame, imgsz=640, conf=0.1)

    # Draw results on the frame
    frame = results[0].plot()  # Annotated frame with bounding boxes

    # Display the frame with detections
    cv2.imshow("Face Detection", frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
