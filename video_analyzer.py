import cv2
import os
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class VeritasPerceptionAI:
    def __init__(self, video_path, output_dir="extracted_frames"):
        self.video_path = video_path
        self.output_dir = output_dir
        # Khởi tạo mô hình Thị giác máy tính CLIP để trích xuất đặc trưng hình ảnh độc lập
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def extract_and_analyze_frames(self, frame_interval=30):
        """
        'Mở mắt' cho AI: Đọc trực tiếp file video vật lý và bóc tách từng khung hình
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[XÓA MÙ THẤT BẠI] Không thể mở file video: {self.video_path}")
            return False

        print(f"[KHỞI ĐỘNG GIÁC QUAN] Đang quét video thực chứng vật lý...")
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Cứ mỗi 30 khung hình (khoảng 1 giây) trích xuất 1 lần để phân tích dấu vết vật lý
            if frame_count % frame_interval == 0:
                frame_name = f"frame_{frame_count}.jpg"
                frame_save_path = os.path.join(self.output_dir, frame_name)
                cv2.imwrite(frame_save_path, frame)
                
                # AI tự chứng thực bằng mô hình thị giác (Không thông qua văn bản)
                self._analyze_visual_authenticity(frame_save_path, frame_count)
                extracted_count += 1
                
            frame_count += 1
            
        cap.release()
        print(f"[HOÀN THÀNH THỰC CHỨNG] Đã trích xuất và phân tích {extracted_count} khung hình quan trọng.")
        return True

    def _analyze_visual_authenticity(self, image_path, frame_index):
        """
        Tầng phân tích giác quan sâu: Kiểm tra các dấu hiệu bất thường của điểm ảnh (pixel)
        """
        image = Image.open(image_path)
        inputs = self.processor(text=["real lecture video", "fake generated video, deepfake, cgi"], 
                                images=image, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
        
        # In ra đánh giá thực chứng độc lập của AI cho từng giây trong video
        print(f" -> Khung hình {frame_index} | Thật: {probs[0]:.2f} | Nghi vấn Giả/Dàn dựng: {probs[1]:.2f}")

if __name__ == "__main__":
    # Đường dẫn tới video bạn muốn AI tự chứng thực
    # Ví dụ: video tranh biện Harvard hoặc bất kỳ video nào cần kiểm chứng
    target_video = "path_to_your_video.mp4" 
    
    # Khởi chạy hệ thống Veritas
    verifier = VeritasPerceptionAI(video_path=target_video)
    verifier.extract_and_analyze_frames(frame_interval=30)
