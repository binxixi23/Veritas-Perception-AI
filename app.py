import streamlit as st
import os
import cv2
from PIL import Image
from video_analyzer import VeritasPerceptionAI

# Configure Page Layout
st.set_page_config(
    page_title="Veritas-Perception-AI Dashboard",
    page_icon="👁️",
    layout="wide"
)

# Header Section
st.title("👁️ Veritas-Perception-AI")
st.subheader("Autonomous Grounding and Sensory Verification Dashboard")
st.markdown("""
> **"Saying 'No' to Indirect Belief."**  
This dashboard forces the system to break down videos into physical matrix layers, rejecting text hallucinations and validating empirical reality directly.
""")

st.divider()

# Sidebar Setup
st.sidebar.header("📁 Media Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload Target Video for Empirical Verification", type=["mp4", "mov", "avi"])
frame_interval = st.sidebar.slider("Sampling Rate (Analyze 1 frame every X frames):", min_value=10, max_value=120, value=30)

# Workspace Layout split into two columns
col1, col2 = st.columns([1, 1])

if uploaded_file is not None:
    # Save video to a temporary local cache
    temp_dir = "temp_inputs"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    video_path = os.path.join(temp_dir, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())
        
    with col1:
        st.info("### 📺 Input Source Stream")
        st.video(video_path)
        
    with col2:
        st.warning("### 🧠 Perceptual Processing Logs")
        run_analysis = st.button("🚀 Initiate Sensory Verification Pipeline")
        
        if run_analysis:
            output_dir = "app_extracted_frames"
            
            # Instantiate your exact processing engine
            with st.spinner("Deconstructing video streams into physical tensors..."):
                verifier = VeritasPerceptionAI(video_path=video_path, output_dir=output_dir)
                
                # Setup custom st.write interceptor or trigger process
                success = verifier.extract_and_analyze_frames(frame_interval=frame_interval)
                
            if success:
                st.success("🎉 Verification Pipeline Complete!")
                st.markdown("#### 🖼️ Inspected Physical Evidence Matrix")
                
                # Render the extracted physical frames dynamically
                files = sorted([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
                if files:
                    # Show up to 4 frames as an explicit gallery grid
                    cols = st.columns(min(len(files), 4))
                    for idx, file_name in enumerate(files[:4]):
                        with cols[idx]:
                            img_path = os.path.join(output_dir, file_name)
                            img = Image.open(img_path)
                            st.image(img, caption=f"Frame: {file_name}", use_column_width=True)
            else:
                st.error("Failed to decouple video feed parameters.")
else:
    with col1:
        st.info("Waiting for a physical video file upload in the sidebar panel...")
    with col2:
        st.write("System Status: **Idle (Awaiting Perceptual Inputs)**")
