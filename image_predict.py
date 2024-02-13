from ultralytics import YOLO
from pathlib import Path


# Load a pretrained YOLOv8n-cls Classify model
model = YOLO('yolov8m-face.pt')

# Run inference on an image
images = Path("D:/CV_FaceDetection/face_det_ds/test/images").glob("*.jpg")  # list of images
images = list(images) # convert from generator to list
print(images)


results = model(images)  # results list

# View results
for r in results:
    print(r.probs)  # print the Probs object containing the detected class probabilities