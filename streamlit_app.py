import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import shutil
import gdown

# URL of the model folder on Google Drive (direct download link)
MODEL_URL = "https://drive.google.com/uc?id=1DRJQilPlTbcGDd40keqtw8CKro_cr5Yc"  #
MODEL_DIR = "model"
EXPORT_PATH = os.path.join(MODEL_DIR, "models")  # Path to the directory where the model will be saved

# Function to download and extract the model folder
def download_and_extract_model(url, export_path):
    if os.path.exists(export_path):
        shutil.rmtree(export_path)  # Remove any existing model directory to avoid conflicts

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Download the model using gdown
    model_zip_path = os.path.join(MODEL_DIR, "model.zip")
    gdown.download(url, model_zip_path, quiet=False, fuzzy=True)

    # Extract the model folder if it is a zip file
    shutil.unpack_archive(model_zip_path, MODEL_DIR)

    # Clean up the zip file
    os.remove(model_zip_path)

# Download and extract the model
download_and_extract_model(MODEL_URL, EXPORT_PATH)

# Load the model
try:
    model = tf.keras.models.load_model(EXPORT_PATH)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Function to preprocess the uploaded image
def preprocess_image(image):
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
    try:
        predictions = model.predict(input_data)
        predicted_class = np.argmax(predictions, axis=1)  # Get the index of the highest probability
        
        # Display the result
        if predicted_class[0] == 0:
            st.write("The model predicts: **No Brain Tumor Detected.**")
        else:
            st.write("The model predicts: **Brain Tumor Detected.**")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
