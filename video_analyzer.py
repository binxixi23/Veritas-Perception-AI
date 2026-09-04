import cv2
import os
import torch
import yt_dlp
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class VeritasPerceptionAI:
    def __init__(self, target_source, output_dir="extracted_frames", temp_input_dir="temp_inputs"):
        # Chuẩn hóa URL: Loại bỏ mốc thời gian (&t=...) gây lỗi phân tích chuỗi
        if isinstance(target_source, str) and "youtube.com" in target_source and "&t=" in target_source:
            target_source = target_source.split("&t=")[0]
            
        self.target_source = target_source
        self.output_dir = output_dir
        self.temp_input_dir = temp_input_dir
        
        # Khởi tạo mô hình mạng thần kinh thị giác sâu (CLIP Transformer)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Tạo hạ tầng thư mục chạy ứng dụng
        for path in [self.output_dir, self.temp_input_dir]:
            if not os.path.exists(path):
                os.makedirs(path)

    def resolve_media_stream(self):
        """
        Cơ chế Thu thập Luồng Linh hoạt: Thâm nhập lớp bảo mật của YouTube, Facebook, TikTok.
        Trả về đường dẫn tệp cục bộ HOẶC đường dẫn luồng trực tiếp (Direct Stream URL).
        """
        ydl_opts = {
            # Thử tải định dạng mp4 chuẩn trước, nếu không được sẽ lấy luồng video có sẵn tốt nhất
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': f'{self.temp_input_dir}/%(title)s.%(ext)s',
            'quiet': True,
            'no_warnings': True,
            # Giả lập trình duyệt phổ thông để tránh bị chặn hệ thống
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'referer': 'https://youtube.com',
        }
        
        print(f"[THÂM NHẬP LUỒNG] Đang kết nối tới nút dữ liệu: {self.target_source}")
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Trích xuất thông tin gốc của video trước khi quyết định tải
                info = ydl.extract_info(self.target_source, download=False)
                
                # PHƯƠNG ÁN 1: Thử tải tệp vật lý về máy local
                try:
                    ydl_opts['format'] = 'best'  # Chuyển sang chế độ kết hợp mượt mà để tải nhanh
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl_downloader:
                        download_info = ydl_downloader.extract_info(self.target_source, download=True)
                        filename = ydl_downloader.prepare_filename(download_info)
                        if os.path.exists(filename):
                            print(f"[INGESTION SUCCESS] Đã tải thành công tệp video gốc: {filename}")
                            return filename
                except Exception as dl_error:
                    print(f"[LƯU Ý HỆ THỐNG] Không thể tải file vật lý do rào cản IP/Quốc gia. Chuyển sang Phương án 2...")
                
                # PHƯƠNG ÁN 2: Giải mã đường dẫn luồng trực tiếp (Direct Stream URL) - Bỏ qua chặn tải file
                if 'url' in info:
                    print(f"[INGESTION SUCCESS] Đã trích xuất luồng trực tiếp (Direct Stream): Khởi chạy bộ giải mã ma trận...")
                    return info['url']
                elif 'formats' in info and len(info['formats']) > 0:
                    # Lấy đường dẫn luồng của định dạng kết hợp video+audio tốt nhất có sẵn
                    valid_formats = [f for f in info['formats'] if f.get('url')]
                    if valid_formats:
                        print(f"[INGESTION SUCCESS] Đã lấy luồng từ danh sách định dạng thay thế...")
                        return valid_formats[-1]['url']
                        
            return None
        except Exception as e:
            print(f"[THẤT BẠI DIỄN TIẾN] Lỗi thâm nhập nền tảng: {str(e)}")
            return None

    def extract_and_analyze_frames(self, frame_interval=30):
        working_target = self.target_source
        
        # Nếu đầu vào là một liên kết mạng, kích hoạt bộ giải mã luồng linh hoạt
        if isinstance(self.target_source, str) and (self.target_source.startswith("http://") or self.target_source.startswith("https://")):
            resolved_stream = self.resolve_media_stream()
            if not resolved_stream:
                return False
            working_target = resolved_stream

        # OpenCV nạp luồng trực tiếp hoặc file cục bộ vào bộ nhớ để cắt khung hình
        cap = cv2.VideoCapture(working_target)
        if not cap.isOpened():
            print(f"[XÓA MÙ THẤT BẠI] Không thể đọc ma trận mảng của video: {working_target}")
            return False

        print(f"[HỢP NHẤT GIÁC QUAN] Đang tiến hành quét ma trận điểm ảnh thực chứng...")
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Trích xuất mẫu phân phối vật lý theo tần suất lấy mẫu được thiết lập
            if frame_count % frame_interval == 0:
                frame_name = f"frame_{frame_count}.jpg"
                frame_save_path = os.path.join(self.output_dir, frame_name)
                cv2.imwrite(frame_save_path, frame)
                
                # Chạy mô hình phân tích thị giác ma trận sâu
                self._analyze_visual_authenticity(frame_save_path, frame_count)
                extracted_count += 1
                
            frame_count += 1
            
        cap.release()
        print(f"[HOÀN THÀNH THỰC CHỨNG] Đã phân tích thành công {extracted_count} cấu trúc tensor sâu.")
        return True

    def _analyze_visual_authenticity(self, image_path, frame_index):
        image = Image.open(image_path)
        inputs = self.processor(
            text=["real authentic footage", "fake generated media, deepfake simulation, staged cgi"], 
            images=image, return_tensors="pt", padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        
        print(f" -> Khung hình Index {frame_index:05d} | Thực chứng Thực tế: {probs[0][0]:.2f} | Xu hướng Sai số: {probs[0][1]:.2f}")

if __name__ == "__main__":
    test_target = "https://youtube.comwatch?v=RBgulEJ2jGw" 
    verifier = VeritasPerceptionAI(target_source=test_target)
    verifier.extract_and_analyze_frames(frame_interval=60)
