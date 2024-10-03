import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import os
import gdown

# URL of the model on Google Drive (direct download link format)
MODEL_URL = "https://drive.google.com/uc?id=1WtREQ7fGoZoEfOTdBL1ptoS30UNN8oUI"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")

# Expected size of the downloaded model file in bytes (check the actual size)
EXPECTED_FILE_SIZE = 85 * 1024 * 1024  # Update this with the actual size in bytes

# Function to download the model
def download_model(url, model_path):
    # Remove existing corrupted file if any
    if os.path.exists(model_path):
        os.remove(model_path)
        
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))

    # Download the model using gdown
    gdown.download(url, model_path, quiet=False, fuzzy=True)

    # Verify the downloaded file size
    if os.path.getsize(model_path) != EXPECTED_FILE_SIZE:
        st.error("The downloaded file size does not match the expected size. Download might be corrupted.")
        return False

    return True

# Download the model
if download_model(MODEL_URL, MODEL_PATH):
    try:
        # Load the model with custom objects
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'KerasLayer': hub.KerasLayer})
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()
else:
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
