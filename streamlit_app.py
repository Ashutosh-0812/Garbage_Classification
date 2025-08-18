import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the Keras model
@st.cache_resource  # Prevents reloading on every run
def load_model():
    return tf.keras.models.load_model("Educell_Garbage.keras", compile=False)
model = load_model()

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
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

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

st.markdown("---")
st.markdown("Made with ❤️ for smart waste segregation")

