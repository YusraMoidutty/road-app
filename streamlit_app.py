import tempfile
import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Page Configuration
st.set_page_config(
    page_title="Road Boundary & 3D EV Perception", page_icon="🛣️", layout="wide"
)

# Custom Styling
st.markdown(
    """
    <style>
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
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return YOLO("best_preprocessed.pt")


model = load_model()

st.title("🛣️ Road Boundary & 3D EV Perception System")
st.write(
    "Detect road boundaries and visualize 3D spatial vehicle perception."
)

st.sidebar.header("Model Settings")
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

# Tabs Navigation
tab1, tab2, tab3 = st.tabs(
    ["📷 Image Detection", "🎥 Video Detection", "🧊 3D Perception View"]
)

# ----------------- TAB 1: IMAGE -----------------
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

# ----------------- TAB 2: VIDEO -----------------
with tab2:
    uploaded_video = st.file_uploader(
        "Upload Road Video", type=["mp4", "avi", "mov"], key="vid_upload"
    )
    if uploaded_video is not None:
        st.video(uploaded_video)
        if st.button("Run Video Detection", key="btn_vid"):
            with st.spinner("Processing video frames..."):
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

# ----------------- TAB 3: 3D PERCEPTION VIEW -----------------
with tab3:
    st.markdown("### 🚘 Interactive 3D Ego-Vehicle & Boundary Scene")
    st.write("Rotate and zoom the 3D canvas below to inspect the trajectory.")

    # Generate 3D Road Plane Data
    x_road = np.linspace(-6, 6, 30)
    y_road = np.linspace(0, 50, 50)
    X, Y = np.meshgrid(x_road, y_road)
    Z = np.zeros_like(X)

    fig = go.Figure()

    # 1. Road Surface Plane
    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale=[[0, "#1e1e1e"], [1, "#2a2a2a"]],
            showscale=False,
            name="Driveable Area",
        )
    )

    # 2. Left Boundary Marker (Red)
    y_line = np.linspace(0, 50, 50)
    fig.add_trace(
        go.Scatter3d(
            x=[-4] * 50,
            y=y_line,
            z=[0.1] * 50,
            mode="lines",
            line=dict(color="#FF3366", width=8),
            name="Left Boundary",
        )
    )

    # 3. Right Boundary Marker (Red)
    fig.add_trace(
        go.Scatter3d(
            x=[4] * 50,
            y=y_line,
            z=[0.1] * 50,
            mode="lines",
            line=dict(color="#FF3366", width=8),
            name="Right Boundary",
        )
    )

    # 4. Host Electric Vehicle (Blue Box at origin)
    fig.add_trace(
        go.Scatter3d(
            x=[0],
            y=[2],
            z=[0.8],
            mode="markers+text",
            marker=dict(size=14, color="#00D2FF", symbol="square"),
            text=["Host EV"],
            textposition="top center",
            name="Ego Vehicle",
        )
    )

    # 5. Detected Surrounding Vehicles (Yellow Boxes ahead)
    fig.add_trace(
        go.Scatter3d(
            x=[-1.8, 2.1],
            y=[18, 32],
            z=[0.8, 0.8],
            mode="markers+text",
            marker=dict(size=12, color="#FFCC00", symbol="diamond"),
            text=["Car 1 (18m)", "Car 2 (32m)"],
            textposition="top center",
            name="Detected Obstacles",
        )
    )

    # 3D Scene Camera & Styling Configuration
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Lateral Offset (m)", backgroundcolor="#0E1117"),
            yaxis=dict(title="Distance Ahead (m)", backgroundcolor="#0E1117"),
            zaxis=dict(title="Height (m)", backgroundcolor="#0E1117"),
            aspectratio=dict(x=1, y=2.5, z=0.5),
            camera=dict(eye=dict(x=0, y=-1.5, z=1.2)),
        ),
        paper_bgcolor="#0E1117",
        margin=dict(l=0, r=0, b=0, t=30),
    )

    st.plotly_chart(fig, use_container_width=True)
