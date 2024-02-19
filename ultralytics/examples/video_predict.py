from ultralytics import YOLO

model = YOLO('D:/CV_FaceDetection/runs/detect/train12/train12/weights/best.pt')
#model = YOLO('yolov8m-face')

results = model(source='sample2.mp4', show=True, conf=0.4, save=True, boxes=True, stream=False)

