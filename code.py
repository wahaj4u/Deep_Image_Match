import os
import requests
import numpy as np
import pandas as pd
import cv2
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

@st.cache_resource
def load_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    return Model(inputs=base_model.input, outputs=base_model.output)
model = load_model()


# App title
st.title("Streamlit Sample App: Image Uploader")

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Open the uploaded image
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Get image details
    st.write("Image Details:")
    st.write(f"Format: {image.format}")
    st.write(f"Size: {image.size}")
    st.write(f"Mode: {image.mode}")

    # Convert the image to a NumPy array and display shape
    img_array = np.array(image)
    st.write(f"Image Shape (NumPy Array): {img_array.shape}")
else:
    st.write("Please upload an image to proceed.")
