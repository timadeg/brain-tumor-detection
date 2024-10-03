import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# URL of the model on Google Drive (direct download link format)
MODEL_URL = "https://drive.google.com/uc?id=1cfFcl_Aric2g-1_GDyldZxE71F3Tou3u"  
MODEL_PATH = "model.h5"

# Function to download the model
def download_model(url, model_path):
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))

    # Download the model using gdown
    gdown.download(url, model_path, quiet=False)

# Download the model
download_model(MODEL_URL, MODEL_PATH)

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Convert image to RGB if it is not
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img = image.resize((224, 224))  # Resize to the input size expected by the model
    img_array = np.array(img) / 255.0  # Normalize pixel values to the [0, 1] range
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension: shape (1, 224, 224, 3)
    return img_array

# Streamlit app
st.title('Brain Tumor Detector')
st.write("Upload an MRI image to detect if a brain tumor is present.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)
    
    # Preprocess the image
    input_data = preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(input_data)  # Directly call the model with the input data
    predicted_class = np.argmax(predictions, axis=1)  # Get the index of the highest probability
    
    # Display the result
    if predicted_class[0] == 0:
        st.write("The model predicts: **No Brain Tumor Detected.**")
    else:
        st.write("The model predicts: **Brain Tumor Detected.**")
