import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
export_path = 'C:/Users/TRITON 500SE/Brain Tumor Detector/models/1727922290/'
model = tf.keras.models.load_model(export_path)

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
    predictions = model.predict(input_data)
    predicted_class = np.argmax(predictions, axis=1)  # Get the index of the highest probability
    
    # Display the result
    if predicted_class[0] == 0:
        st.write("The model predicts: **No Brain Tumor Detected.**")
    else:
        st.write("The model predicts: **Brain Tumor Detected.**")
