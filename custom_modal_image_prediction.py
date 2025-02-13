from ultralytics import YOLO

# change this path
model = YOLO('/home/biswash/Documents/yolo_biswash/lane_segmentation/best.pt')


results = model.predict("picture1.jpg", imgsz=640, conf=0.1, save=True, show=True)
