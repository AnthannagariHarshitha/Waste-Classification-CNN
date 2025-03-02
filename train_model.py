import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ✅ Ensure Streamlit settings are at the top
st.set_page_config(page_title="Waste Classification", layout="centered")

# ✅ Define model loading function
@st.cache_resource
def load_classification_model():
    model_path = "waste_classification_model.h5"
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at {model_path}. Please upload or train the model.")
        return None
    return load_model(model_path)

# ✅ Load model
model = load_classification_model()

# ✅ Define class labels
CLASS_NAMES = ['Glass', 'Paper', 'Cardboard', 'Plastic', 'Metal', 'Trash']

# ✅ Preprocessing function
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, (150, 150))  # Resize for CNN model
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize pixel values
    return image

# ✅ Streamlit UI
st.title("♻️ Waste Classification Using CNN")
st.write("Upload an image to classify waste material.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# ✅ Image Prediction
if uploaded_file is not None:
    if model is None:
        st.warning("⚠️ Model not loaded. Please check the file path.")
    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)  # Decode image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess and Predict
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # ✅ Display Results
        st.subheader(f"Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}%")
        st.success("✅ Thank you for keeping the environment clean!")

        # Debugging: Show Raw Predictions
        st.write("🔍 Raw Predictions:", prediction)
else:
    st.warning("📂 Please upload an image for classification.")
