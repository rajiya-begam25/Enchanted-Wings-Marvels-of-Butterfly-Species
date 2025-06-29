import cv2
import numpy as np
from src.preprocess import IMG_SIZE

def predict_image(model, img_path, label_map):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    inv_label_map = {v: k for k, v in label_map.items()}
    return inv_label_map[class_index]
