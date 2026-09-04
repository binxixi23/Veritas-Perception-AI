import streamlit as st
import os
import cv2
import glob
from PIL import Image
from video_analyzer import VeritasPerceptionAI

# Configure Page Layout
st.set_page_config(
    page_title="Veritas-Perception-AI Dashboard",
    page_icon="👁️",
    layout="wide"
)

# Custom styling to isolate operational controls
st.markdown("""
    <style>
    .reportview-container { background: #0f1116; }
    .stButton>button { width: 100%; border-radius: 4px; background-color: #1f2937; color: white; }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.title("👁️ Veritas-Perception-AI")
st.subheader("Autonomous Grounding and Sensory Verification Dashboard")
st.markdown("""
> **"Saying 'No' to Indirect Belief."**  
This dashboard intercepts raw digital media vectors—via local file uploads or direct streaming source nodes—and subjects them to autonomous high-dimensional pixel validation.
""")

st.divider()

# Sidebar Configuration
st.sidebar.header("📁 Media Control Panel")

# Select Input Vector Type
source_type = st.sidebar.radio("Select Media Source Vector Type:", ["Direct YouTube URL", "Local Video File"])
frame_interval = st.sidebar.slider("Sampling Rate (Analyze 1 frame every X frames):", min_value=10, max_value=120, value=30)

target_source = None
is_ready = False

if source_type == "Direct YouTube URL":
    youtube_url = st.sidebar.text_input("Paste Direct YouTube URL:", placeholder="https://youtube.com...")
    if youtube_url:
        target_source = youtube_url
        is_ready = True
else:
    uploaded_file = st.sidebar.file_uploader("Upload Target Video File", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        # Cache physical upload locally
        temp_dir = "temp_inputs"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        target_source = os.path.join(temp_dir, uploaded_file.name)
        with open(target_source, "wb") as f:
            f.write(uploaded_file.read())
        is_ready = True

# Workspace Layout
col1, col2 = st.columns([1, 1])

if is_ready and target_source:
    with col1:
        st.info("### 📺 Input Source Stream Managed")
        if source_type == "Direct YouTube URL":
            st.success(f"Target Link Logged: **{target_source}**")
            st.caption("The engine will extract binary streams from this endpoint during execution.")
        else:
            st.video(target_source)
        
    with col2:
        st.warning("### 🧠 Perceptual Processing Logs")
        run_analysis = st.button("🚀 Initiate Sensory Verification Pipeline")
        
        if run_analysis:
            output_dir = "app_extracted_frames"
            
            # Clear previous frame artifacts to keep evaluations pure
            if os.path.exists(output_dir):
                files = glob.glob(f"{output_dir}/*")
                for f in files:
                    try:
                        os.remove(f)
                    except:
                        pass

            # Execute the empirical validation mechanism
            with st.spinner("Deconstructing target vector streams into physical matrix structures..."):
                verifier = VeritasPerceptionAI(target_source=target_source, output_dir=output_dir)
                success = verifier.extract_and_analyze_frames(frame_interval=frame_interval)
                
            if success:
                st.success("🎉 Verification Pipeline Complete!")
                st.markdown("#### 🖼️ Inspected Physical Evidence Matrix")
                
                # Fetch output imagery
                extracted_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
                if extracted_files:
                    # Dynamically construct up to a 4-column display grid
                    display_count = min(len(extracted_files), 4)
                    cols = st.columns(display_count)
                    for idx, file_name in enumerate(extracted_files[:display_count]):
                        with cols[idx]:
                            img_path = os.path.join(output_dir, file_name)
                            img = Image.open(img_path)
                            st.image(img, caption=f"Matrix State: {file_name}", use_column_width=True)
                else:
                    st.info("Deconstruction executed safely, but no persistent frame frames were exported.")
            else:
                st.error("[SFL ERROR] Failed to decouple stream parameters from target source endpoint.")
else:
    with col1:
        st.info("Waiting for a physical file upload or a live network URL pointer...")
    with col2:
        st.write("System Status: **Idle (Awaiting Perceptual Inputs)**")
