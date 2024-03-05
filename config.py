# fps_estimator_config
from pathlib import Path

device = 'cuda'  # device selection for predictions cuda/cpu
model_dir = Path('F:\CV_FaceDetection\Models\FER\yolov8x-FER.pt') # absolute path of the model being used for predictions
download_models = False  # whether to download pretrained model from gdrive or not, set False after initial run


