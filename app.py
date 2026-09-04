import streamlit as st
import os
import cv2
import glob
from PIL import Image
from video_analyzer import VeritasPerceptionAI

# Cấu hình giao diện trang
st.set_page_config(
    page_title="Veritas-Perception-AI Dashboard",
    page_icon="👁️",
    layout="wide"
)

# Giao diện nền tối chuyên sâu cho phòng thí nghiệm khoa học máy tính
st.markdown("""
    <style>
    .reportview-container { background: #0f1116; }
    .stButton>button { width: 100%; border-radius: 4px; background-color: #1f2937; color: white; }
    </style>
""", unsafe_allow_html=True)

# Phần tiêu đề dự án
st.title("👁️ Veritas-Perception-AI")
st.subheader("Autonomous Grounding and Sensory Verification Dashboard")
st.markdown("""
> **"Saying 'No' to Indirect Belief."**  
Giao diện thực chứng vạn năng này trực tiếp đánh chặn các vector truyền tải đa phương thức, bỏ qua các diễn giải văn bản để kiểm tra trực tiếp cấu trúc vật lý của ma trận điểm ảnh.
""")

st.divider()

# Cấu hình bảng điều khiển ở Sidebar
st.sidebar.header("📁 Bảng điều khiển luồng")

# Lựa chọn nguồn dữ liệu đầu vào
source_type = st.sidebar.radio("Chọn dạng Vector nguồn đầu vào:", ["Universal Stream Link (Mọi nền tảng)", "Local Video File"])
frame_interval = st.sidebar.slider("Tần suất lấy mẫu (Quét 1 khung hình sau mỗi X khung):", min_value=10, max_value=120, value=30)

target_source = None
is_ready = False

if source_type == "Universal Stream Link (Mọi nền tảng)":
    stream_url = st.sidebar.text_input(
        "Nhập URL liên kết video công khai:", 
        placeholder="Hỗ trợ YouTube, TikTok, Facebook, Vimeo, hoặc link trực tiếp .mp4..."
    )
    if stream_url:
        target_source = stream_url
        is_ready = True
else:
    uploaded_file = st.sidebar.file_uploader("Tải lên file video cục bộ", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        # Lưu file tải lên vào bộ nhớ đệm tạm thời
        temp_dir = "temp_inputs"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        target_source = os.path.join(temp_dir, uploaded_file.name)
        with open(target_source, "wb") as f:
            f.write(uploaded_file.read())
        is_ready = True

# Phân chia bố cục khu vực làm việc (2 cột chính xác)
col1, col2 = st.columns(2)

if is_ready and target_source:
    with col1:
        st.info("### 📺 Đã ghi nhận dòng dữ liệu đầu vào")
        if source_type == "Universal Stream Link (Mọi nền tảng)":
            st.success(f"Điểm cuối tài nguyên (URL): **{target_source}**")
            st.caption("Tầng hợp nhất giác quan (SFL) sẽ tự động thâm nhập và phân tích luồng dữ liệu nhị phân từ địa chỉ mạng này.")
        else:
            st.video(target_source)
        
    with col2:
        st.warning("### 🧠 Nhật ký xử lý của Giác quan AI")
        run_analysis = st.button("🚀 Khởi chạy Tầng Hợp nhất Giác quan (SFL)")
        
        if run_analysis:
            output_dir = "app_extracted_frames"
            
            # Xóa các tệp ảnh phân tích cũ để đảm bảo kết quả thực chứng độc lập
            if os.path.exists(output_dir):
                files = glob.glob(f"{output_dir}/*")
                for f in files:
                    try:
                        os.remove(f)
                    except:
                        pass

            # Khởi chạy đường truyền thực chứng phổ quát
            with st.spinner("Đang bóc tách và bẻ gãy cấu trúc video thành các ma trận điểm ảnh..."):
                verifier = VeritasPerceptionAI(target_source=target_source, output_dir=output_dir)
                success = verifier.extract_and_analyze_frames(frame_interval=frame_interval)
                
            if success:
                st.success("🎉 Quá trình thực chứng vật lý hoàn tất!")
                st.markdown("#### 🖼️ Ma trận bằng chứng vật lý đã kiểm chứng")
                
                # Lấy danh sách ảnh đã trích xuất thành công
                extracted_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
                if extracted_files:
                    # Tạo lưới hiển thị động tối đa 4 cột ảnh bằng chứng
                    display_count = min(len(extracted_files), 4)
                    cols = st.columns(display_count)
                    for idx, file_name in enumerate(extracted_files[:display_count]):
                        with cols[idx]:
                            img_path = os.path.join(output_dir, file_name)
                            img = Image.open(img_path)
                            st.image(img, caption=f"Trạng thái: {file_name}", use_column_width=True)
                else:
                    st.info("Quá trình giải mã an toàn, nhưng không có tệp ảnh tensor nào được lưu lại.")
            else:
                st.error("[SFL ERROR] Thất bại trong việc đánh chặn gói tin truyền tải từ địa chỉ mạng được chỉ định. Vui lòng kiểm tra lại kết nối hoặc link video.")
else:
    with col1:
        st.info("Đang chờ một file luồng vật lý hoặc một con trỏ liên kết mạng được gán...")
    with col2:
        st.write("Trạng thái hệ thống: **Đang rảnh (Chờ đợi dữ liệu giác quan)**")
