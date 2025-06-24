# import streamlit as st
# import numpy as np
# import tensorflow as tf
# import cv2
# import os
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array

# # ✅ Streamlit Page Configuration
# st.set_page_config(page_title="Waste Classification", layout="wide")

# # ✅ Load Classification Model
# @st.cache_resource
# def load_classification_model():
#     model_path = "waste_classification_model.h5"
#     if not os.path.exists(model_path):
#         st.error(f"❌ Model file not found at {model_path}. Please upload or train the model.")
#         return None
#     return load_model(model_path)

# model = load_classification_model()

# # ✅ Define Class Labels
# CLASS_NAMES = ['Glass', 'Paper', 'Cardboard', 'Plastic', 'Metal', 'Trash']

# # ✅ Preprocess Image Function
# def preprocess_image(image):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
#     image = cv2.resize(image, (150, 150))  # Resize for CNN model
#     image = img_to_array(image)
#     image = np.expand_dims(image, axis=0)  # Add batch dimension
#     image = image / 255.0  # Normalize pixel values
#     return image

# # ✅ Sidebar Navigation
# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Go to:", ["Home", "About"])

# if page == "Home":
#     st.title("♻️ Waste Classification Using CNN")
#     st.write("Upload or capture an image to classify waste material.")
    
#     # ✅ Upload Image Option
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
#     # ✅ Webcam Capture
#     use_camera = st.checkbox("Use Webcam")
#     if use_camera:
#         captured_image = st.camera_input("Take a picture")
#         if captured_image:
#             uploaded_file = captured_image
    
#     # ✅ Image Prediction
#     if uploaded_file is not None:
#         if model is None:
#             st.warning("⚠️ Model not loaded. Please check the file path.")
#         else:
#             file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#             image = cv2.imdecode(file_bytes, 1)  # Decode image
#             st.image(image, caption="Uploaded Image", use_column_width=True)

#             # Preprocess and Predict
#             processed_image = preprocess_image(image)
#             prediction = model.predict(processed_image)
#             predicted_class = CLASS_NAMES[np.argmax(prediction)]
#             confidence = np.max(prediction) * 100

#             # ✅ Display Results
#             st.subheader(f"Prediction: {predicted_class}")
#             st.write(f"Confidence: {confidence:.2f}%")
#             st.success("✅ Thank you for keeping the environment clean!")

#             # Debugging: Show Raw Predictions
#             with st.expander("🔍 View Raw Predictions"):
#                 st.write(prediction)

# elif page == "About":
#     st.title("📖 About This App")
#     st.write("This waste classification app uses a CNN model to classify waste into different categories. Upload an image or take a picture to predict.")


import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# App title
st.set_page_config(page_title="Waste Classifier", page_icon="♻️")
st.title("♻️ Waste Classification using CNN")
st.write("Upload a waste image to classify it as **Biodegradable** or **Non-Biodegradable**.")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()

# Define labels (update if your model uses different ones)
class_names = ["Biodegradable", "Non-Biodegradable"]

# Prediction function
def predict_image(img):
    image = img.resize((64, 64))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape((1, 64, 64, 3))
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    return predicted_class, confidence

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Predicting...")
    label, confidence = predict_image(image)
    st.success(f"Prediction: **{label}** ({confidence:.2f}% confidence)")

