import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Constants
IMAGE_SIZE = 128
TESTING_DIR = "dataset/test"
MODEL_NAME = "models/cnn.vgg16.keras"

def load_or_train_model() -> tf.keras.models.Model:
    """Load an existing model or train a new one."""
    try:
        return load_model(MODEL_NAME)
    except Exception as e:
        print(f"Model not found. Error: {e}")
        return None

def predict_class(image_path: str, model: tf.keras.models.Model) -> str:
    """Predict class name for an input image."""
    if model is None:
        return "Oops! It's not your fault."

    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image:", image_path)
        return None

    img = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype("float32") / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    return os.path.basename(os.path.dirname(image_path))

def main():
    model = load_or_train_model()
    if not model:
        print("Could not load model.")
        return
    
    results = []
    for root, dirs, files in os.walk(TESTING_DIR):
        for file in files:
            image_path = os.path.join(root, file)
            prediction = predict_class(image_path, model)
            if prediction is not None:
                results.append(f"\n{image_path}\n-> {prediction}")
    
    print("\n[Results]")
    for result in results:
        print(result)

if __name__ == "__main__":
    main()