import os
import torch
from pathlib import Path
import tensorflow as tf
    
def _download_model():
    print("DETA: Downloading Landmark Estimation Model...")
    from deta import Deta
    deta = Deta(os.environ.get("DETA_ID"))
    models = deta.Drive("models")
    model_file = models.get('scoliovis_segmentation_model.h5')
    with open("models/scoliovis_segmentation_model.h5", "wb+") as f:
        for chunk in model_file.iter_chunks(1024):
            f.write(chunk)
        print("DETA: Landmark Estimation Model downloaded.")
        model_file.close()

def _download_detection_model():
    print("DETA: Downloading Object Detection Model...")
    from deta import Deta
    deta = Deta(os.environ.get("DETA_ID"))
    models = deta.Drive("models")
    model_file = models.get('detection_model.pt')
    with open("models/detection_model.pt", "wb+") as f:
        for chunk in model_file.iter_chunks(1024):
            f.write(chunk)
        print("DETA: Object Detection model downloaded.")
        model_file.close()

def get_detection_model():
    model_path = Path("models/detection_model.pt")

    # Download if the model does not exist
    if model_path.is_file():
        print("Detection Model is already downloaded.")
    else:
        print("Detection Model was NOT FOUND.")
        _download_detection_model()
    
    # Get model from path and return
    model = torch.hub.load('./yolov5', 'custom', path='./models/detection_model.pt', source='local')
    return model
    
def get_model():
    model_path = Path('models/scoliovis_segmentation_model.h5')
    if model_path.is_file():
        print("The model is already downloaded.")
    else:
        print("The model is NOT downloaded yet.")
        _download_model()
    
    model = tf.keras.models.load_model('models/scoliovis_segmentation_model.h5')
    return model
        