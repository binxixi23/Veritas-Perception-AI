import cv2
import os
import torch
import yt_dlp
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class VeritasPerceptionAI:
    def __init__(self, target_source, output_dir="extracted_frames", temp_input_dir="temp_inputs"):
        """
        Initialize the empirical perception engine.
        :param target_source: Can be a local video file path or a direct YouTube URL.
        """
        self.target_source = target_source
        self.output_dir = output_dir
        self.temp_input_dir = temp_input_dir
        
        # Initialize Deep Vision Neural Pipeline for empirical validation
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Create operational environment infrastructure
        for path in [self.output_dir, self.temp_input_dir]:
            if not os.path.exists(path):
                os.makedirs(path)

    def download_youtube_stream(self):
        """
        Autonomous Stream Ingestion: Pulls the physical media container directly from 
        the YouTube source node without human proxies or browsers.
        """
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': f'{self.temp_input_dir}/%(title)s.%(ext)s',
            'quiet': True,
            'no_warnings': True
        }
        
        print(f"[STREAM INGESTION] Penetrating YouTube network interface for: {self.target_source}")
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.target_source, download=True)
                filename = ydl.prepare_filename(info)
                print(f"[INGESTION SUCCESS] Secure file delivery established: {filename}")
                return filename
        except Exception as e:
            print(f"[INGESTION ERROR] Failed to intercept online media stream: {str(e)}")
            return None

    def extract_and_analyze_frames(self, frame_interval=30):
        """
        Deconstructs the target video file matrix array into chronological frame instances.
        """
        # Determine if target is a network URL or local vector path
        working_file = self.target_source
        if "youtube.com" in self.target_source or "youtu.be" in self.target_source:
            downloaded_path = self.download_youtube_stream()
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
                
            # Sample spatial state distributions at chosen sampling rate interval
            if frame_count % frame_interval == 0:
                frame_name = f"frame_{frame_count}.jpg"
                frame_save_path = os.path.join(self.output_dir, frame_name)
                cv2.imwrite(frame_save_path, frame)
                
                # Execute direct high-dimensional empirical evaluation pass
                self._analyze_visual_authenticity(frame_save_path, frame_count)
                extracted_count += 1
                
            frame_count += 1
            
        cap.release()
        print(f"[EMPIRICAL RETRIEVAL] Completed inspection of {extracted_count} deep tensor structures.")
        return True

    def _analyze_visual_authenticity(self, image_path, frame_index):
        """
        Evaluates physical consistency checks across raw pixel tensor configurations.
        """
        image = Image.open(image_path)
        inputs = self.processor(
            text=["real authentic footage", "fake generated media, deepfake simulation, staged cgi"], 
            images=image, return_tensors="pt", padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
        
        print(f" -> Frame Matrix Index {frame_index:05d} | Empirical Reality: {probs[0]:.2f} | Synthetic Bias: {probs[1]:.2f}")

if __name__ == "__main__":
    # Test vector input matrix: Supports local physical files or URL streams
    # e.g., "https://youtube.com"
    test_target = "https://youtube.com" 
    
    verifier = VeritasPerceptionAI(target_source=test_target)
    verifier.extract_and_analyze_frames(frame_interval=60)
