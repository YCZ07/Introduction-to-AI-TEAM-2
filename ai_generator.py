# ai_generator.py
from PIL import Image, ImageDraw

class ImageGenerationModel:
    def __init__(self):
        print("[AI Model] 임시 테스트용 더미 모델 로드 완료")
        self.is_ready = True

    def generate_art(self, spectrogram_image):
        """
        스펙트로그램 이미지를 받아 소리에 맞는 그림을 생성하는 함수 (현재는 더미)
        """
        print("[AI Model] 스펙트로그램 기반 이미지 생성 중 (더미)...")
        # 원본 스펙트로그램과 동일한 크기의 캔버스 생성
        dummy_result = Image.new('RGB', spectrogram_image.size, color=(70, 130, 180))
        
        # 가짜 결과물임을 알 수 있게 텍스트 추가
        draw = ImageDraw.Draw(dummy_result)
        draw.text((20, 20), "AI Generated Art (Temp)", fill=(255, 255, 255))
        
        return dummy_result