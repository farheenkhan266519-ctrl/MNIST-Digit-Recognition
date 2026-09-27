import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
    /* Force a clean light background across the whole app */
    .stApp {
        background-color: #ffffff;
        color: #000000;
    }

    .main {
        padding-top: 1rem;
        background-color: #ffffff;
    }

    .block-container {
        padding-top: 3rem !important;
    }

    /* Make every default text element black for strong contrast */
    h1, h2, h3, h4, h5, h6, p, span, label, li, div {
        color: #000000;
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
        color: #000000;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
        color: #333333;
    }

    .result-box {
        padding: 10px;
        text-align: center;
        color: #000000;
    }

    .result-box h2 {
        color: #000000;
    }

    .digit {
        font-size: 70px;
        font-weight: bold;
        margin: 10px;
        color: #1a1a1a;
    }

    .confidence {
        font-size: 22px;
        font-weight: 600;
        color: #000000;
    }

    .info-card {
        padding: 18px;
        border-radius: 12px;
        margin-top: 20px;
        background-color: #f0f0f0;
        border: 1px solid #dddddd;
        color: #000000;
    }

    /* Buttons: keep them clearly visible with black text on light background */
    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-size: 18px;
        font-weight: 600;
        background-color: #ffffff;
        color: #000000;
        border: 2px solid #000000;
    }

    div.stButton > button:hover {
        background-color: #000000;
        color: #ffffff;
        border: 2px solid #000000;
    }

    /* File uploader box - remove background, show only the button */
    section[data-testid="stFileUploader"],
    div[data-testid^="stFileUploader"],
    section[data-testid="stFileUploader"] div {
        background-color: transparent !important;
        background: none !important;
        border: none !important;
        box-shadow: none !important;
    }

    section[data-testid="stFileUploader"] {
        padding: 0;
    }

    div[data-testid="stFileUploaderDropzone"] {
        border: none !important;
    }

    section[data-testid="stFileUploader"] * {
        color: #000000 !important;
    }

    /* Upload button - parrot green */
    button[data-testid="stBaseButton-secondary"] {
        background-color: #6CC24A !important;
        border: 2px solid #4E9A31 !important;
        color: #000000 !important;
    }

    button[data-testid="stBaseButton-secondary"]:hover {
        background-color: #7ED957 !important;
        border: 2px solid #3F7A26 !important;
    }

    /* Expander header/body text */
    .streamlit-expanderHeader {
        color: #000000 !important;
        background-color: #f5f5f5;
    }

    .streamlit-expanderContent {
        background-color: #ffffff;
        color: #000000;
    }

    /* Success/info boxes: keep readable dark text on light backgrounds */
    div[data-testid="stAlert"] {
        color: #000000;
    }

    /* Push the uploaded image down slightly */
    div[data-testid="stImage"] {
        margin-top: 20px;
    }

    /* Hide the fullscreen expand icon on images */
    button[title="View fullscreen"] {
        display: none !important;
    }

    div[data-testid="stElementToolbar"] {
        display: none !important;
    }

    /* Divider color */
    hr {
        border-color: #dddddd;
    }

    /* Hide the default Streamlit top decoration bar */
    div[data-testid="stDecoration"] {
        display: none;
    }

    /* Make the top header/toolbar white instead of dark */
    header[data-testid="stHeader"] {
        background-color: #ffffff;
        box-shadow: none;
    }

    div[data-testid="stToolbar"] {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("mnist_random_forest_model.pkl")


model = load_model()

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="title">🔢 MNIST Digit Recognition</div>',
    unsafe_allow_html=True
)

st.divider()

# -----------------------------
# Image Upload
# -----------------------------
st.subheader("📤 Upload a Digit Image")

uploaded_file = st.file_uploader(
    "Choose an image containing a handwritten digit",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Load image
    image = Image.open(uploaded_file)

    # -----------------------------
    # Process Image
    # -----------------------------
    gray_image = ImageOps.grayscale(image)

    # Resize to MNIST format
    resized_image = gray_image.resize((28, 28))

    # Convert to array
    image_array = np.array(resized_image)

    # Invert image if background is brighter than digit
    if image_array.mean() > 127:
        image_array = 255 - image_array

    # Normalize
    image_array = image_array / 255.0

    # Flatten to 784 features
    image_array = image_array.reshape(1, 784)

    # -----------------------------
    # Prediction (runs automatically, shown side-by-side with image)
    # -----------------------------
    prediction = model.predict(image_array)[0]

    probabilities = model.predict_proba(image_array)[0]

    confidence = probabilities[int(prediction)] * 100

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.image(
                image,
                caption="Uploaded Image",
                width=250
            )

    with col2:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="result-box">
                    <h2>🎯 Prediction Result</h2>
                    <div class="digit">{prediction}</div>
                    <div class="confidence">
                        Confidence: {confidence:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.success(
                f"The model predicted the digit **{prediction}**."
            )