import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile

st.set_page_config(page_title="Road Boundary Detector", layout="wide")
st.title("🛣️ Road Boundary Detection System")

@st.cache_resource
def load_model():
    # Loads your custom YOLO weights file
    return YOLO("best_preprocessed.pt")

model = load_model()

# Sidebar configuration controls
st.sidebar.header("Settings")
mode = st.sidebar.radio("Select Input Mode", ["Image Detection", "Video Detection"])
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

if mode == "Image Detection":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
            
        if st.button("Run Detection"):
            with st.spinner("Processing image..."):
                results = model.predict(source=image, conf=conf_thresh)
                res_plotted = results[0].plot()
                with col2:
                    st.image(res_plotted[..., ::-1], caption="Detected Boundaries", use_container_width=True)

elif mode == "Video Detection":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        st.video(uploaded_video)
        if st.button("Process Video"):
            with st.spinner("Processing video frames..."):
                # Store uploaded video to temp directory for OpenCV processing
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_video.read())
                
                cap = cv2.VideoCapture(tfile.name)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                output_path = "processed_output.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results = model.predict(source=frame, conf=conf_thresh, verbose=False)
                    out.write(results[0].plot())
                    
                cap.release()
                out.release()
                
                st.success("Processing complete!")
                st.video(output_path)
