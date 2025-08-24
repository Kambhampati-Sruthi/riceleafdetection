import streamlit as st
import numpy as np
import joblib
import json
import base64
from datetime import datetime
from gtts import gTTS
from skimage.io import imread
from skimage.transform import resize
# Load model & preprocessors
svm = joblib.load("models/svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")
pca = joblib.load("models/pca.pkl")
with open("models/label_map.json", "r") as f:
    class_map = json.load(f)

# Precaution messages in English and Telugu
precautions = {
    "healthy": {
        "English": "No issues detected. Maintain proper watering and nutrition.",
        "Telugu": "ఎలాంటి సమస్యలు కనిపించలేదు. సరైన నీటిపారుదల మరియు పోషకాహారాన్ని కొనసాగించండి."
    },
    "healthy_rice_leaf": {
        "English": "Rice leaf is healthy. Continue regular monitoring.",
        "Telugu": "బియ్యం ఆకులు ఆరోగ్యంగా ఉన్నాయి. క్రమం తప్పకుండా పరిశీలన కొనసాగించండి."
    },
    "rice_brown_spot": {
        "English": "Apply fungicides, avoid excessive nitrogen, and use resistant varieties.",
        "Telugu": "ఫంగిసైడ్స్‌ను ఉపయోగించండి, అధిక నైట్రోజన్‌ను నివారించండి, మరియు ప్రతిఘటించే రకాలను ఉపయోగించండి."
    }
}

color_map = {
    "healthy": "green",
    "healthy_rice_leaf": "orange",
    "rice_brown_spot": "red"
}

IMG_SIZE = 128

def predict_image(img):
    img_resized = resize(img, (IMG_SIZE, IMG_SIZE)).flatten().reshape(1, -1)
    img_scaled = scaler.transform(img_resized)
    img_pca = pca.transform(img_scaled)
    probs = svm.predict_proba(img_pca)[0]
    pred_class = np.argmax(probs)
    return class_map[str(pred_class)], probs[pred_class]

def generate_download_link(label, conf, precaution):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = f"""🌾 Mini Plantix - Rice Disease Report 🌾

Prediction: {label.replace('_', ' ').title()}
Confidence: {conf:.2f}
Time: {timestamp}

📋 Precaution Advice:
{precaution}

Thank you for using Mini Plantix. Keep your crops healthy!
"""
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{label}_report.txt">📥 Download Precaution Report</a>'
    return href

def speak_precaution(text, lang_choice):
    lang_code = "en" if lang_choice == "English" else "te"
    tts = gTTS(text, lang=lang_code)
    tts.save("precaution.mp3")
    audio_file = open("precaution.mp3", "rb")
    st.audio(audio_file.read(), format="audio/mp3")

# Streamlit UI
st.set_page_config(page_title="Mini Plantix", layout="centered")
st.title("🌱 Mini Plantix - Rice Disease Detection")
st.write("Upload a leaf image or use camera to detect disease and get precautions.")

voice_lang = st.selectbox("🔈 Choose voice language", ["English", "Telugu"])
option = st.radio("Choose input method:", ["Upload Image", "Camera Input"])

if "history" not in st.session_state:
    st.session_state.history = []

def process_image(img):
    label, conf = predict_image(img)
    st.image(img, caption=f"Prediction: {label} ({conf:.2f} confidence)", use_column_width=True)

    st.markdown(
        f"<div style='background-color:{color_map[label]};padding:10px;border-radius:5px;color:white;text-align:center;'>"
        f"<b>Status:</b> {label.replace('_', ' ').title()}</div>",
        unsafe_allow_html=True
    )

    precaution_text = precautions[label][voice_lang]
    st.success(f"Precaution: {precaution_text}")
    speak_precaution(precaution_text, voice_lang)
    st.markdown(generate_download_link(label, conf, precaution_text), unsafe_allow_html=True)
    st.session_state.history.append((label, precaution_text))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = imread(uploaded_file)
        process_image(img)

elif option == "Camera Input":
    camera_image = st.camera_input("Take a picture")
    if camera_image:
        img = imread(camera_image)
        process_image(img)

with st.expander("📜 View Prediction History"):
    for i, (cls, prec) in enumerate(st.session_state.history):
        st.write(f"{i+1}. **{cls.replace('_', ' ').title()}** → {prec}")
