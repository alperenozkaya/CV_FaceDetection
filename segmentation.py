from ultralytics import YOLO

model = YOLO('D:/CV_FaceDetection/runs\detect/train10/train10/weights/best.pt')

results = model(source='sample2.mp4', show=True, conf=0.4, save=True, boxes=True, stream=False)