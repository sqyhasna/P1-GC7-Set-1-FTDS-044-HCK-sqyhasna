import streamlit as st
import numpy as np
import tensorflow as tf
import os
import gdown

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16

# Load Model
@st.cache_resource
def load_model():
    # URL Google Drive file
    drive_url = "https://drive.google.com/file/d/1P45mwsUGe2PmETvZFIrZCzttwSm_OHNO/view?usp=sharing"
    local_path = "model_coffee_vgg16.keras"

    # Download hanya kalau file belum ada
    if not os.path.exists(local_path):
        gdown.download(drive_url, local_path, quiet=False)

    model = tf.keras.models.load_model(local_path)
    return model

model = load_model()

def prediction(image):
    image = image.resize((224, 224))
    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input_vgg16(image_array)

    y_pred_proba = model.predict(image_array)
    y_pred_class = np.argmax(y_pred_proba, axis=1)[0]

    class_names = [
        'Dark',
        'Green',
        'Light',
        'Medium'
    ]

    return class_names[y_pred_class]


def run():
    st.title('Coffee Bean Classification')

    st.write(
        'Upload gambar coffee bean untuk mengetahui tingkat roasting '
        'berdasarkan class Dark, Green, Light, atau Medium.'
    )

    uploaded_file = st.file_uploader(
        'Upload Image',
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file is not None:
        image = tf.keras.utils.load_img(uploaded_file)

        st.image(
            image,
            caption='Uploaded Image',
            use_container_width=True
        )

        if st.button('Predict'):
            result = prediction(image)

            st.subheader('Prediction Result')
            st.write(result)

if __name__ == "__main__":
    run()