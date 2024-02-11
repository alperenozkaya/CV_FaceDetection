from ultralytics import YOLO

model = YOLO('face_det_v8n_detect_epochs.pt')

results = model(source='sample1.mp4', show=True, save=True)
# webcam
#results = model(source=0, show=True, save=False)