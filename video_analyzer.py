import cv2
import os
import torch
import yt_dlp
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class VeritasPerceptionAI:
    def __init__(self, target_source, output_dir="extracted_frames", temp_input_dir="temp_inputs"):
        self.target_source = target_source
        self.output_dir = output_dir
        self.temp_input_dir = temp_input_dir
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        for path in [self.output_dir, self.temp_input_dir]:
            if not os.path.exists(path):
                os.makedirs(path)

    def download_universal_stream(self):
        """
        Universal Ingestion Pipeline: Intercepts physical video containers from 
        Facebook, TikTok, Vimeo, or direct cloud endpoints using yt-dlp.
        """
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': f'{self.temp_input_dir}/%(title)s.%(ext)s',
            'quiet': True,
            'no_warnings': True,
            # Bypasses bot detection algorithms on strict platforms like TikTok/Facebook
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        print(f"[UNIVERSAL INGESTION] Penetrating target media node stream: {self.target_source}")
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.target_source, download=True)
                filename = ydl.prepare_filename(info)
                print(f"[INGESTION SUCCESS] Universal payload safely delivered: {filename}")
                return filename
        except Exception as e:
            print(f"[INGESTION ERROR] Target platform stream acquisition blocked: {str(e)}")
            return None

    def extract_and_analyze_frames(self, frame_interval=30):
        working_file = self.target_source
        
        # UNIVERSAL PATCH: Check if source is ANY online URL link
        if self.target_source.startswith("http://") or self.target_source.startswith("https://"):
            downloaded_path = self.download_universal_stream()
            if not downloaded_path:
                return False
            working_file = downloaded_path

        cap = cv2.VideoCapture(working_file)
        if not cap.isOpened():
            print(f"[XÓA MÙ THẤT BẠI] Unable to map video matrix array: {working_file}")
            return False

        print(f"[SENSORY FUSION] Initiating continuous physical pixel matrix scan...")
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frame_name = f"frame_{frame_count}.jpg"
                frame_save_path = os.path.join(self.output_dir, frame_name)
                cv2.imwrite(frame_save_path, frame)
                
                self._analyze_visual_authenticity(frame_save_path, frame_count)
                extracted_count += 1
                
            frame_count += 1
            
        cap.release()
        print(f"[EMPIRICAL RETRIEVAL] Completed inspection of {extracted_count} deep tensor structures.")
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
        
        print(f" -> Frame Matrix Index {frame_index:05d} | Empirical Reality: {probs[0][0]:.2f} | Synthetic Bias: {probs[0][1]:.2f}")

if __name__ == "__main__":
    # Test vector endpoint (Can accept any valid media platform URL link)
    test_target = "https://vimeo.com" 
    verifier = VeritasPerceptionAI(target_source=test_target)
    verifier.extract_and_analyze_frames(frame_interval=60)
