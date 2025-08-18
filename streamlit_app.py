import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Debugging: Print TensorFlow version
st.write(f"TensorFlow Version: {tf.__version__}")

# Clear TensorFlow session to avoid name scope issues
tf.keras.backend.clear_session()

# Load the Keras model
@st.cache_resource  # Prevents reloading on every run
def load_model():
    try:
        model = tf.keras.models.load_model("Educell_Garbage.keras", compile=False)
        st.success("Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Check if model loaded successfully
if model is None:
    st.error("Failed to load the model. Please check the model file and TensorFlow version.")
    st.stop()

# Define class names as per your training
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

st.title("♻️ Garbage Classification AI")
st.write(
    "Upload an image of waste and the AI will classify it as **Plastic**, **Glass**, **Paper**, etc."
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocessing (resizing, scaling)
    img_resized = img.resize((124, 124))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    try:
        # Prediction
        prediction = model.predict(img_array)[0]
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction)) * 100

        st.markdown(
            f"""
            ### 📁 **Predicted Class:** `{predicted_class.upper()}`
            ### 🎯 **Confidence:** `{confidence:.2f}%`
            """
        )
    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.markdown("---")
st.markdown("Made with ❤️ for smart waste segregation")