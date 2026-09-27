import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps

st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 MNIST Digit Recognition")
st.write("Upload a handwritten digit image and let the model recognize it.")

@st.cache_resource
def load_model():
    return joblib.load("mnist_random_forest_model.pkl")

model = load_model()

st.success("Random Forest model loaded successfully!")

uploaded_file = st.file_uploader(
    "📤 Upload a digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")

    st.subheader("Uploaded Image")
    st.image(image, width=250)

    # Convert image to 28 × 28
    image = ImageOps.invert(image)
    image = image.resize((28, 28))

    # Convert image to NumPy array
    image_array = np.array(image)

    # Normalize pixels
    image_array = image_array / 255.0

    # Flatten to 784 features
    image_array = image_array.reshape(1, 784)

    # Prediction
    prediction = model.predict(image_array)[0]

    # Confidence
    probabilities = model.predict_proba(image_array)[0]
    confidence = np.max(probabilities) * 100

    st.markdown("---")

    st.subheader("Prediction Result")

    st.success(f"🔢 Predicted Digit: {prediction}")

    st.info(f"🎯 Confidence: {confidence:.2f}%")

st.markdown("---")

st.caption("MNIST Digit Recognition | Task 2 | Farheen Khan")