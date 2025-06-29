import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

IMG_SIZE = 128

def load_data(data_dir):
    categories = os.listdir(data_dir)
    label_map = {cat: i for i, cat in enumerate(categories)}

    X, y = [], []
    for category in categories:
        path = os.path.join(data_dir, category)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                X.append(img)
                y.append(label_map[category])
            except:
                pass
    return np.array(X)/255.0, to_categorical(np.array(y)), label_map
