import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# -------- CONFIG --------
MODEL_FILENAME = "improved_cnn_model.h5"

# -------- CLASS LABELS --------
CLASS_NAMES = [
    'champa', 'bili', 'pipal', 'kesudo', 'shirish', 'bamboo', 'other', 'motichanoti',
    'khajur', 'gunda', 'cactus', 'mango', 'gulmohar', 'jamun', 'banyan', 'saptaparni',
    'neem', 'sonmahor', 'babul', 'pilikaren', 'asopalav', 'vad', 'nilgiri', 'kanchan',
    'sitafal', 'sugarcane', 'simlo', 'garmalo', 'amla', 'coconut'
]

# -------- UI: Sidebar --------
st.sidebar.title("🌿 Tree Classifier Settings")
st.sidebar.write("Use this app to classify tree species based on leaf or bark images.")
st.sidebar.info("Make sure the model file improved_cnn_model.h5 is in your app folder.")

# -------- LOAD MODEL (CACHED) --------
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILENAME):
        try:
            st.sidebar.success("✅ Model loaded successfully.")
            return tf.keras.models.load_model(MODEL_FILENAME)
        except Exception as e:
            st.sidebar.error(f"❌ Could not load model: {e}")
            return None
    else:
        st.sidebar.error("📁 Model file not found.")
        return None

# -------- PREPROCESS IMAGE --------
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# -------- MAIN PAGE --------
st.markdown("<h1 style='text-align: center;'>🌳 Tree Species Classification</h1>", unsafe_allow_html=True)
st.write("Upload a leaf or bark image below, and I’ll try to predict which tree it belongs to!")

model = load_model()

uploaded_file = st.sidebar.file_uploader("📸 Upload image here", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Your uploaded image", use_column_width=True)

    if model:
        with st.spinner("🔍 Analyzing..."):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            predicted_index = np.argmax(prediction)
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = prediction[0][predicted_index]

        # Display prediction result
        st.success(f"🌱 Predicted Tree Species: *{predicted_class}* with {confidence:.2%} confidence")

        # Display full prediction probabilities
        st.subheader("📊 Prediction Confidence for All Classes:")
        confidences = [(CLASS_NAMES[i], float(prediction[0][i])) for i in range(len(CLASS_NAMES))]
        confidences.sort(key=lambda x: x[1], reverse=True)
        for name, score in confidences:
            st.write(f"{name}: {score:.2%}")

    else:
        st.warning("⚠ Model not available.")
else:
    st.info("👈 Upload an image in the sidebar to get started.")