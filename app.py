import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests
import zipfile

# URL of the model on GitHub (raw content URL or a direct download link)
MODEL_URL = "https://github.com/timadeg/brain-tumor-detection/raw/main/path/to/model.zip"  # Replace with your model's URL

# Directory to save the downloaded model
MODEL_DIR = "model"

# Function to download and extract the model
def download_and_extract_model(url, model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Download the model
    model_zip_path = os.path.join(model_dir, "model.zip")
    response = requests.get(url)
    with open(model_zip_path, "wb") as f:
        f.write(response.content)

    # Extract the model if it is a zip file
    with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
        zip_ref.extractall(model_dir)

    # Clean up the zip file
    os.remove(model_zip_path)

# Download and extract the model
download_and_extract_model(MODEL_URL, MODEL_DIR)

# Load the model using TFSMLayer
export_path = os.path.join(MODEL_DIR, "model_directory")  # Replace with the extracted model directory name
model = tf.keras.Sequential([
    tf.keras.layers.TFSMLayer(export_path, call_endpoint='serving_default')
])

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
    predictions = model(input_data)  # Directly call the model with the input data
    predicted_class = np.argmax(predictions, axis=1)  # Get the index of the highest probability
    
    # Display the result
    if predicted_class[0] == 0:
        st.write("The model predicts: **No Brain Tumor Detected.**")
    else:
        st.write("The model predicts: **Brain Tumor Detected.**")
