from ultralytics import YOLO

model = YOLO('F:/CV_FaceDetection/runs/detect/train26/weights/best.pt')

results = model(source='sample1.mp4', show=True, conf=0.4, save=False, boxes=True, stream=False)