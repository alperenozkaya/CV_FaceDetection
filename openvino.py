# openvino is not a package error persists

from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.export(format='openvino')

ov_model = YOLO('yolov8n_openvino_model/')

results_ov = ov_model('https://ultralytics.com/images/bus.jpg')

results_original = model('https://ultralytics.com/images/bus.jpg')
