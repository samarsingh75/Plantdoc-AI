import sys
import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

MODEL_PATH = os.environ.get('MODEL_PATH', '../saved_model/plant_disease_model.h5')
IMAGE_SIZE = (128, 128)

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize(IMAGE_SIZE)
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def main():
    st.title("Image classifier Deep learning")
    st.markdown("Upload a plant leaf image to detect the disease.")

    uploaded_file = st.file_uploader("Upload the Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=False, width=300)

        if st.button("PREDICT"):
            with st.spinner("Predicting..."):
                model = load_model()
                img_array = preprocess_image(image)
                predictions = model.predict(img_array)
                result_index = int(np.argmax(predictions[0]))
                predicted_class = CLASS_NAMES[result_index]
                confidence = float(predictions[0][result_index]) * 100

            st.markdown("**result...**")
            st.write(f"Predicted label : **{predicted_class}**")
            st.write(f"Confidence : **{confidence:.1f}**")


if __name__ == '__main__':
    main()
