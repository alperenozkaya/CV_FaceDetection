from ultralytics import YOLO


def main():
    model = YOLO('yolov8n.pt')

    # Train the model
    model.train(data='data.yaml', epochs=100, imgsz=640, batch=8)


if __name__ == '__main__':
    main()