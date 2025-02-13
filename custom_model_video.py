from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('best.pt')

# Path to the video file
video_path = 'lane_basketball.mp4'  # Replace with your video file path

# Open the video file
cap = cv2.VideoCapture(video_path)


# Check if the video-file is opened
if not cap.isOpened():
    print("Error: Could not open Video.")
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
