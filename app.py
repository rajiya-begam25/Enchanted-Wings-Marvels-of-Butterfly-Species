import streamlit as st
from PIL import Image
import numpy as np
import os
from tensorflow.keras.models import load_model
from src.predict import predict_image
from src.preprocess import IMG_SIZE

st.title("🦋 Enchanted Wings: Butterfly Species Classifier")

uploaded_file = st.file_uploader("Upload a butterfly image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_path = "temp.jpg"
    image.save(img_path)

    model = load_model("model/butterfly_model.h5")

    # Assume label_map is saved as a dictionary in a .npy file
    import numpy as np
    label_map = np.load("model/label_map.npy", allow_pickle=True).item()

    label = predict_image(model, img_path, label_map)

    st.subheader("Predicted Species:")
    st.success(label)
