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
            # Custom spinner notification encouraging interactive 3D inspection while waiting
            with st.spinner(
                "⏳ Processing road boundaries... Please be patient, it will upload soon! Meanwhile, check out our interactive 3D Perception View in Tab 3 above..Dont not press simulated EV motion while proessing the video! 🚘"
            ):
                results = model.predict(source=image, conf=conf_thresh)
                res_plotted = results[0].plot()
                with col2:
                    st.markdown("**Processed Boundary Output**")
                    st.image(
                        res_plotted[..., ::-1], use_container_width=True
                    )

# ----------------- TAB 2: VIDEO DETECTION -----------------
with tab2:
    st.markdown("### 🎥 Road Driving Video Inference")
    uploaded_video = st.file_uploader(
        "Upload Road Driving Video", type=["mp4", "avi", "mov"], key="vid_upload"
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("Run Video Detection", key="btn_vid"):
            with st.spinner(
                "⏳ Processing video frames... Please be patient! Meanwhile, feel free to inspect our 3D Perception View tab above! 🎥"
            ):
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

    # Driveable Surface
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
            z=[0.1] * 60,
            mode="lines",
            line=dict(color="#FF3366", width=8),
            name="Left Boundary",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[4.5] * 60,
            y=y_line,
            z=[0.1] * 60,
            mode="lines",
            line=dict(color="#FF3366", width=8),
            name="Right Boundary",
        )
    )

    # Center Lane Dashed Divider
    fig.add_trace(
        go.Scatter3d(
            x=[0] * 60,
            y=y_line,
            z=[0.05] * 60,
            mode="lines",
            line=dict(color="#FFCC00", width=4, dash="dash"),
            name="Center Lane",
        )
    )

    # Ego EV Position
    fig.add_trace(
        go.Scatter3d(
            x=[0],
            y=[ev_speed],
            z=[0.8],
            mode="markers+text",
            marker=dict(size=16, color="#00D2FF", symbol="square"),
            text=["⚡ Ego EV"],
            textposition="top center",
            name="Host EV",
        )
    )

    # Surrounding Vehicles
    fig.add_trace(
        go.Scatter3d(
            x=[-2, 2.2],
            y=[ev_speed + 15, ev_speed + 30],
            z=[0.8, 0.8],
            mode="markers+text",
            marker=dict(size=13, color="#FF9900", symbol="diamond"),
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
            camera=dict(eye=dict(x=0, y=-1.3, z=1.1)),
        ),
        paper_bgcolor="#0E1117",
        height=500,
        margin=dict(l=0, r=0, b=0, t=20),
    )

    st.plotly_chart(fig, use_container_width=True)
