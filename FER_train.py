from ultralytics import YOLO


def main():
    model = YOLO('F:/CV_FaceDetection/runs/detect/train23/weights/best.pt') # pretrained face detection model

    # Train the model
    model.train(data='FER_conf.yaml', epochs=100, imgsz=640, batch=8)


if __name__ == '__main__':
    main()