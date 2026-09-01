import os
import subprocess
import tempfile
import time
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

# Tabs Navigation
tab1, tab2, tab3 = st.tabs(
    ["🧊 3D EV Perception View", "📷 Image Detection", "🎥 Video Detection"]
)


def create_vehicle_mesh(x_center, y_center, z_center, color, name):
    """Generates 3D box meshes representing detailed vehicle body & roof."""
    w, l, h = 1.8, 4.0, 1.2

    # Lower Chassis Vertices
    x = [
        x_center - w / 2,
        x_center + w / 2,
        x_center + w / 2,
        x_center - w / 2,
        x_center - w / 2,
        x_center + w / 2,
        x_center + w / 2,
        x_center - w / 2,
    ]
    y = [
        y_center - l / 2,
        y_center - l / 2,
        y_center + l / 2,
        y_center + l / 2,
        y_center - l / 2,
        y_center - l / 2,
        y_center + l / 2,
        y_center + l / 2,
    ]
    z = [
        z_center,
        z_center,
        z_center,
        z_center,
        z_center + h,
        z_center + h,
        z_center + h,
        z_center + h,
    ]

    i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
    j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
    k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]

    mesh_body = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        color=color,
        opacity=0.9,
        name=name,
        flatshading=True,
    )

    # Upper Cabin/Roof Vertices
    rw, rl, rh = 1.4, 2.0, 0.8
    rx = [
        x_center - rw / 2,
        x_center + rw / 2,
        x_center + rw / 2,
        x_center - rw / 2,
        x_center - rw / 2,
        x_center + rw / 2,
        x_center + rw / 2,
        x_center - rw / 2,
    ]
    ry = [
        y_center - rl / 2,
        y_center - rl / 2,
        y_center + rl / 2,
        y_center + rl / 2,
        y_center - rl / 2,
        y_center - rl / 2,
        y_center + rl / 2,
        y_center + rl / 2,
    ]
    rz = [
        z_center + h,
        z_center + h,
        z_center + h,
        z_center + h,
        z_center + h + rh,
        z_center + h + rh,
        z_center + h + rh,
        z_center + h + rh,
    ]

    mesh_roof = go.Mesh3d(
        x=rx,
        y=ry,
        z=rz,
        i=i,
        j=j,
        k=k,
        color="#222222",
        opacity=0.8,
        showlegend=False,
        flatshading=True,
    )

    return [mesh_body, mesh_roof]


def build_3d_scene(pos_y):
    """Generates the Plotly 3D Figure centered on the host vehicle at pos_y."""
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

    # Center Lane Divider
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

    # Ego Host EV (Cyan 3D Mesh)
    ego_traces = create_vehicle_mesh(
        0, pos_y, 0.1, "#00D2FF", "⚡ Ego EV (Host)"
    )
    for trace in ego_traces:
        fig.add_trace(trace)

    # Surrounding Traffic Vehicles
    v1_traces = create_vehicle_mesh(
        -2.2, pos_y + 16, 0.1, "#FF9900", "🚘 Traffic Car 1"
    )
    for trace in v1_traces:
        fig.add_trace(trace)

    v2_traces = create_vehicle_mesh(
        2.2, pos_y + 28, 0.1, "#00FF66", "🚘 Traffic Car 2"
    )
    for trace in v2_traces:
        fig.add_trace(trace)

    # Centered Camera Angle
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title="Lateral (m)", range=[-6, 6], backgroundcolor="#0E1117"
            ),
            yaxis=dict(
                title="Distance Ahead (m)",
                range=[0, 60],
                backgroundcolor="#0E1117",
            ),
            zaxis=dict(
                title="Elevation (m)", range=[0, 8], backgroundcolor="#0E1117"
            ),
            aspectratio=dict(x=1, y=2.5, z=0.5),
            camera=dict(
                eye=dict(x=0, y=-1.8, z=1.4), center=dict(x=0, y=0.2, z=0)
            ),
        ),
        paper_bgcolor="#0E1117",
        height=550,
        margin=dict(l=0, r=0, b=0, t=20),
    )
    return fig


# ----------------- TAB 1: 3D EV PERCEPTION VIEW -----------------
with tab1:
    st.markdown("### 🧊 Live 3D Autonomous EV Navigation View")

    col_btn, col_slider = st.columns([1, 2])
    with col_btn:
        start_anim = st.button("▶ Start 3D Vehicle Animation")
    with col_slider:
        ev_speed = st.slider(
            "Manual EV Depth Controller",
            min_value=5,
            max_value=25,
            value=10,
            help="Adjust to move the EV manually or click the animation button.",
        )

    plot_holder = st.empty()

    if start_anim:
        # Loop animation driving forward through the scene
        for step in range(5, 26, 1):
            fig = build_3d_scene(step)
            plot_holder.plotly_chart(fig, use_container_width=True)
            time.sleep(0.08)
    else:
        fig = build_3d_scene(ev_speed)
        plot_holder.plotly_chart(fig, use_container_width=True)

# ----------------- TAB 2: IMAGE DETECTION -----------------
with tab2:
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
            with st.spinner(
                "⏳ Processing road boundaries... Please be patient, it will upload soon! Meanwhile, check out our interactive 3D Perception View in Tab 1 above! 🚘"
            ):
                results = model.predict(source=image, conf=conf_thresh)
                res_plotted = results[0].plot()
                with col2:
                    st.markdown("**Processed Boundary Output**")
                    st.image(
                        res_plotted[..., ::-1], use_container_width=True
                    )

# ----------------- TAB 3: VIDEO DETECTION -----------------
with tab3:
    st.markdown("### 🎥 Road Driving Video Inference")
    uploaded_video = st.file_uploader(
        "Upload Road Driving Video", type=["mp4", "avi", "mov"], key="vid_upload"
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("Run Video Detection", key="btn_vid"):
            with st.spinner(
                "⏳ Processing video frames... Please be patient, it will upload soon! Meanwhile, check out our interactive 3D Perception View in Tab 1 above! 🎥"
            ):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_video.read())

                cap = cv2.VideoCapture(tfile.name)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                raw_output_path = "temp_raw_output.mp4"
                web_output_path = "processed_output.mp4"

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(
                    raw_output_path, fourcc, fps, (width, height)
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

                cmd = f"ffmpeg -y -i {raw_output_path} -vcodec libx264 {web_output_path}"
                subprocess.run(cmd, shell=True, check=True)

                st.success("Video processing complete!")
                st.video(web_output_path)
