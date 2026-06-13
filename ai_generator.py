# ai_generator.py
import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image

class ImageGenerationModel:
    def __init__(self):
        print("[AI Model] 실제 AI 모델 가중치를 다운로드 및 로드하는 중... (시간이 걸릴 수 있습니다)")
        
        # Stable Diffusion 1.5 모델 로드 (가장 가볍고 범용적)
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_safetensors=True
        )
        
        # 그래픽카드(GPU)가 있다면 GPU를 사용하고, 없으면 CPU 사용
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe.to(self.device)
        print(f"[AI Model] 로드 완료! 사용 기기: {self.device}")

    def generate_art(self, spectrogram_image):
        """
        스펙트로그램 이미지를 바탕으로 실제 그림을 생성합니다.
        """
        print("[AI Model] 스펙트로그램 기반 이미지 생성 중...")
        
        # AI 모델이 요구하는 기본 RGB 포맷으로 확실히 변환
        init_image = spectrogram_image.convert("RGB")
        
        # AI에게 어떤 느낌으로 그릴지 명령(프롬프트) 부여
        # 이 부분을 소리의 특징에 맞게 바꾸면 다양한 결과가 나옵니다.
        prompt = "abstract digital art, colorful sound waves, synesthesia, masterpiece, highly detailed"
        
        # strength(0.0 ~ 1.0): 1에 가까울수록 원본(스펙트로그램)을 무시하고 새로 그림
        # guidance_scale: 프롬프트를 얼마나 강하게 따를 것인지 (보통 7~8)
        result = self.pipe(
            prompt=prompt, 
            image=init_image, 
            strength=0.75, 
            guidance_scale=7.5
        ).images[0]
        
        return result