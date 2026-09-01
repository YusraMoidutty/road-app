import gc
import os
import subprocess
import tempfile
import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Page Configuration
st.set_page_config(
    page_title="EV Road Boundary Perception", page_icon="⚡", layout="wide"
)

# Custom Styling
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


@st.cache_resource
def load_model():
    # Make sure best_preprocessed.pt is in your root directory or update path accordingly
    return YOLO("best_preprocessed.pt")


model = load_model()

st.title("⚡ Electric Vehicle Road & Boundary Perception System")
st.write(
    "Real-time autonomous navigation, boundary detection, and dynamic 3D spatial modeling."
)

# Sidebar Options
st.sidebar.header("Model Settings")
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

# Separate Tabs Navigation
tab1, tab2, tab3 = st.tabs(
    ["📷 Image Detection", "🎥 Video Detection", "🧊 3D EV Perception View"]
)

# ----------------- TAB 1: IMAGE DETECTION -----------------
with tab1:
    st.markdown("### 📷 Road Image Boundary Detection")
    uploaded_image = st.file_uploader(
        "Choose a road photo...", type=["jpg", "jpeg", "png"], key="img_upload"
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original Input**")
            st.image(image, use_container_width=True)

        if st.button("Run Image Boundary Detection", key="btn_img"):
            with st.spinner("⏳ Processing road boundaries..."):
                results = model.predict(source=image, conf=conf_thresh)
                res_plotted = results[0].plot()
                with col2:
                    st.markdown("**Processed Boundary Output**")
                    st.image(res_plotted[..., ::-1], use_container_width=True)

# ----------------- TAB 2: VIDEO DETECTION (MEMORY-SAFE) -----------------
with tab2:
    st.markdown("### 🎥 Road Driving Video Inference")
    uploaded_video = st.file_uploader(
        "Upload Road Driving Video", type=["mp4", "avi", "mov"], key="vid_upload"
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("Run Video Detection", key="btn_vid"):
            with st.spinner("⏳ Processing video efficiently..."):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_video.read())

                cap = cv2.VideoCapture(tfile.name)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                raw_output_path = "temp_raw_output.mp4"
                web_output_path = "processed_output.mp4"

                # Reduce output FPS overhead to save processing power
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(raw_output_path, fourcc, 15, (width, height))

                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    # Skip every 2nd frame to cut RAM and CPU load by 50%
                    if frame_count % 2 != 0:
                        continue

                    # imgsz=320 keeps tensor memory usage low
                    results = model.predict(
                        source=frame, conf=conf_thresh, imgsz=320, verbose=False
                    )

                    out.write(results[0].plot())

                    # Clean up memory buffers explicitly
                    del results
                    if frame_count % 10 == 0:
                        gc.collect()

                cap.release()
                out.release()

                # Low-memory ffmpeg conversion flags
                cmd = f"ffmpeg -y -i {raw_output_path} -vcodec libx264 -preset ultrafast -crf 28 {web_output_path}"
                subprocess.run(cmd, shell=True, check=True)

                st.success("Video processing complete!")
                st.video(web_output_path)

                # Clean temporary raw video files
                if os.path.exists(tfile.name):
                    os.remove(tfile.name)
                if os.path.exists(raw_output_path):
                    os.remove(raw_output_path)

# ----------------- TAB 3: 3D EV PERCEPTION VIEW -----------------
with tab3:
    st.markdown("### 🧊 Live 3D Autonomous EV Navigation View")

    ev_speed = st.slider(
        "Simulated EV Motion Depth",
        min_value=0,
        max_value=20,
        value=10,
        help="Adjust to simulate EV forward movement in 3D space.",
    )

    x_road = np.linspace(-6, 6, 40)
    y_road = np.linspace(0, 60, 60)
    X, Y = np.meshgrid(x_road, y_road)
    Z = np.zeros_like(X)

    fig = go.Figure()

    # Ground surface plane
    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale=[[0, "#12161f"], [1, "#1e2530"]],
            showscale=False,
        )
    )

    # Road Boundary Lines
    y_line = np.linspace(0, 60, 60)
    fig.add_trace(
        go.Scatter3d(
            x=[-4.5] * 60,
            y=y_line,
            z=[0.05] * 60,
            mode="lines",
            line=dict(color="#FF3366", width=8),
            name="Left Boundary",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[4.5] * 60,
            y=y_line,
            z=[0.05] * 60,
            mode="lines",
            line=dict(color="#FF3366", width=8),
            name="Right Boundary",
        )
    )

    # Center Lane Dashed Line
    fig.add_trace(
        go.Scatter3d(
            x=[0] * 60,
            y=y_line,
            z=[0.03] * 60,
            mode="lines",
            line=dict(color="#FFCC00", width=4, dash="dash"),
            name="Center Lane",
        )
    )

    cy = ev_speed

    # Ego EV Node Marker
    fig.add_trace(
        go.Scatter3d(
            x=[0],
            y=[cy],
            z=[0.5],
            mode="markers+text",
            marker=dict(size=10, color="#00D2FF"),
            text=["⚡ Ego EV"],
            textposition="top center",
            name="Ego EV Position",
        )
    )

    # Sensor Field Rays
    sensor_x = [0, -3.5, 3.5, 0]
    sensor_y = [cy, cy + 20, cy + 20, cy]
    sensor_z = [0.5, 0.1, 0.1, 0.5]

    fig.add_trace(
        go.Scatter3d(
            x=sensor_x,
            y=sensor_y,
            z=sensor_z,
            mode="lines",
            line=dict(color="#00FFFF", width=4),
            name="Sensor Field",
        )
    )

    fig.add_trace(
        go.Mesh3d(
            x=[0, -3.5, 3.5],
            y=[cy, cy + 20, cy + 20],
            z=[0.5, 0.1, 0.1],
            color="#00FFFF",
            opacity=0.2,
            name="Detection Field",
        )
    )

    # Surrounding Vehicles
    fig.add_trace(
        go.Scatter3d(
            x=[-2, 2.2],
            y=[ev_speed + 15, ev_speed + 30],
            z=[0.8, 0.8],
            mode="markers+text",
            marker=dict(size=14, color="#FF9900", symbol="diamond"),
            text=["Vehicle 1", "Vehicle 2"],
            textposition="top center",
            name="Ahead Traffic",
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Lateral (m)", backgroundcolor="#0E1117"),
            yaxis=dict(title="Distance Ahead (m)", backgroundcolor="#0E1117"),
            zaxis=dict(title="Elevation (m)", backgroundcolor="#0E1117"),
            aspectratio=dict(x=1, y=2.5, z=0.5),
            camera=dict(eye=dict(x=-0.8, y=-1.5, z=1.0)),
        ),
        paper_bgcolor="#0E1117",
        height=550,
        margin=dict(l=0, r=0, b=0, t=20),
    )

    st.plotly_chart(fig, use_container_width=True)
