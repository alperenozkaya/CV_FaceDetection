from ultralytics import YOLO


# TODO: train yolov8n-m-x 200-300 epochs
# TODO: train without pretrained model
# TODO: different face datasets(wider-face compatible with yolov8 format)
# TODO: search datasets for facial expressions
#model = YOLO('D:/CV_FaceDetection/runs/detect/train12/train12/weights/best.pt')
model = YOLO('yolov8m-face.pt')

results = model(source='sample2.mp4', show=True, conf=0.4, save=True, boxes=True, stream=False)

