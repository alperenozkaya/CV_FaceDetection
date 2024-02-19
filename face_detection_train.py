from ultralytics import YOLO

<<<<<<< HEAD

def main():
    model = YOLO('yolov8n.pt')

    # Train the model
    model.train(data='data.yaml', epochs=100, imgsz=640, batch=8)


if __name__ == '__main__':
    main()
=======
# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='coco128.yaml', epochs=100, imgsz=640)
>>>>>>> 6fbf35bab9927c046499f86ecf5790e25fe87d4f
