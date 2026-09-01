import tempfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Page Configuration
st.set_page_config(
    page_title="Road Boundary Detector", page_icon="🛣️", layout="wide"
)

# Custom CSS Styling
st.markdown(
    """
    <style>
    /* Change button colors and rounded corners */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    /* Hide the default Streamlit hamburger menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""",
    unsafe_allow_html=True,
)


# Cache YOLO model so it loads only once into memory
@st.cache_resource
def load_model():
    return YOLO("best_preprocessed.pt")


model = load_model()

# Header Section
st.title("🛣️ Road Boundary Detection System")
st.write(
    "Upload images or video clips to detect road boundaries in real-time."
)

# Confidence Slider Control
st.sidebar.header("Model Settings")
conf_thresh = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05,
)

# Tabbed Interface Layout
tab1, tab2 = st.tabs(["📷 Image Detection", "🎥 Video Detection"])

# ----------------- TAB 1: IMAGE DETECTION -----------------
with tab1:
    uploaded_image = st.file_uploader(
        "Upload Road Image", type=["jpg", "jpeg", "png"], key="img_upload"
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Original Input")
            st.image(image, use_container_width=True)

        if st.button("Run Image Detection", key="btn_img"):
            with st.spinner("Analyzing image..."):
                results = model.predict(source=image, conf=conf_thresh)
                res_plotted = results[0].plot()

                with col2:
                    st.markdown("### Processed Output")
                    st.image(
                        res_plotted[..., ::-1], use_container_width=True
                    )

# ----------------- TAB 2: VIDEO DETECTION -----------------
with tab2:
    uploaded_video = st.file_uploader(
        "Upload Road Video", type=["mp4", "avi", "mov"], key="vid_upload"
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("Run Video Detection", key="btn_vid"):
            with st.spinner("Processing video frames... Please wait."):
                # Save uploaded file to a temporary location
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_video.read())

                cap = cv2.VideoCapture(tfile.name)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                output_path = "processed_output.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(
                    output_path, fourcc, fps, (width, height)
                )

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results = model.predict(
                        source=frame, conf=conf_thresh, verbose=False
                    )
                    out.write(results[0].plot())

                cap.release()
                out.release()

                st.success("Video processing complete!")
                st.video(output_path)
