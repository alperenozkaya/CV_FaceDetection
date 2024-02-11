from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')

results = model(source=0, show=True, conf=0.4, save=True, boxes=True, stream=False)