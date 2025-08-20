import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load Model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("Educell_Garbage.keras", compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Dark Mode CSS
custom_css = """
<style>
    body, .stApp {
        background-color: #1f1f1f;
        color: #f2f2f2;
        font-family: 'Segoe UI', sans-serif;
    }
    .stMarkdown, .stText, .stFileUploader label {
        color: #f2f2f2 !important;
    }
    .prediction-box {
        background-color: #2b2b2b;
        color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        font-size: 18px;
        font-weight: 500;
    }
    h1, h2, h3 {
        text-align: center;
        color: #00ffcc !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Title
st.title("♻️ Garbage Classification AI")
st.write("Upload an image of waste and the AI will classify it as **Plastic**, **Glass**, **Paper**, etc.")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = img.resize((124, 124))
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    try:
        prediction = model.predict(img_array)[0]
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction)) * 100

        st.markdown(
            f"""
            <div class="prediction-box">
                📁 <b>Predicted Class:</b> {predicted_class.upper()} <br>
                🎯 <b>Confidence:</b> {confidence:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.markdown("---")
st.markdown("Made with ❤️ for smart waste segregation")
