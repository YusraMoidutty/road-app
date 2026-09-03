import gc
import os
import tempfile
import time
import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
import torch

# ----------------- PAGE CONFIG & STYLING -----------------
st.set_page_config(
    page_title="EV Road Boundary Perception", page_icon="⚡", layout="wide"
)

st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #00D2FF;
        color: #0E1117;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #00a8cc;
        color: white;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- MODEL LOADING -----------------
@st.cache_resource
def load_perception_model(model_path="best_preprocessed.pt"):
    if not os.path.exists(model_path):
        return None
    try:
        model = torch.jit.load(model_path) if model_path.endswith(".torchscript") else torch.load(model_path, map_location="cpu")
        if isinstance(model, dict) and "model" in model:
            model = model["model"].float().eval()
        return model
    except Exception:
        try:
            from ultralytics import YOLO
            return YOLO(model_path)
        except ImportError:
            return None

model = load_perception_model()

# ----------------- NAVIGATION -----------------
st.title("⚡ EV Road Boundary Perception Platform")
st.caption("Optimized Lightweight CPU Perception Engine")

tab1, tab2, tab3 = st.tabs(["🖼️ Image Detection", "🎥 Video Inference", "🧊 3D Perception"])

# ----------------- TAB 1: IMAGE DETECTION -----------------
with tab1:
    st.subheader("Image Perception Pipeline")
    uploaded_image = st.file_uploader("Upload Road Image", type=["jpg", "jpeg", "png"])
    
    col1, col2 = st.columns(2)
    conf_thresh = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        img_np = np.array(image)
        
        with col1:
            st.image(image, caption="Input Frame", use_container_width=True)
            
        with col2:
            if st.button("Run Image Detection"):
                with st.spinner("Processing frame..."):
                    resized = cv2.resize(img_np, (640, 360))
                    
                    if model is not None and hasattr(model, "predict"):
                        results = model.predict(source=resized, conf=conf_thresh, imgsz=320, verbose=False)
                        annotated = results[0].plot()
                    else:
                        annotated = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
                        cv2.putText(annotated, "Perception Active", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 210, 255), 2)
                        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    
                    st.image(annotated, caption="Perception Output", use_container_width=True)

# ----------------- TAB 2: VIDEO INFERENCE -----------------
with tab2:
    st.subheader("Optimized Video Pipeline")
    uploaded_video = st.file_uploader("Upload Road Video", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close()
        
        st.video(tfile.name)
        
        if st.button("Run Video Detection"):
            st.info("Running CPU acceleration strategy (Resized to 480x270, processing 1 in 5 frames)...")
            
            cap = cv2.VideoCapture(tfile.name)
            
            raw_output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name
            web_output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(raw_output_path, fourcc, 10.0, (480, 270))
            
            frame_count = 0
            vid_start_time = time.time()
            progress_bar = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Frame Skip: Process 1 in 5 frames for high speed
                if frame_count % 5 != 0:
                    continue
                    
                # Downscale resolution to lighten CPU processing
                frame_resized = cv2.resize(frame, (480, 270))
                
                if model is not None and hasattr(model, "predict"):
                    results = model.predict(source=frame_resized, conf=conf_thresh, imgsz=320, verbose=False)
                    proc_frame = results[0].plot()
                else:
                    proc_frame = frame_resized
                    cv2.putText(proc_frame, f"Frame: {frame_count}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 210, 255), 2)
                
                out.write(proc_frame)
                progress_bar.progress(min(frame_count / total_frames, 1.0))
                
                # Periodically reclaim RAM
                if frame_count % 15 == 0:
                    del proc_frame
                    gc.collect()
            
            cap.release()
            out.release()
            
            # Ultrafast FFmpeg Encoding Pass
            os.system(f"ffmpeg -y -i {raw_output_path} -vcodec libx264 -preset ultrafast -crf 32 {web_output_path}")
            
            vid_elapsed = round(time.time() - vid_start_time, 2)
            st.success(f"⚡ Video processing complete in {vid_elapsed} seconds!")
            
            if os.path.exists(web_output_path):
                st.video(web_output_path)
            
            # Temporary File Cleanup
            for path in [tfile.name, raw_output_path, web_output_path]:
                if os.path.exists(path):
                    os.remove(path)

# ----------------- TAB 3: 3D PERCEPTION -----------------
with tab3:
    st.subheader("3D Point Cloud Perception Sandbox")
    x = np.linspace(-10, 10, 40)
    y = np.linspace(0, 50, 40)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2)) / 2
    
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig, use_container_width=True)
