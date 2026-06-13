# part2_ai/ai_model.py
from PIL import Image
import os

class SoundToImageAI:
    def __init__(self, model_path=None):
        # 1. 여기에서 AI 모델(무게추 파일 등)을 로드합니다.
        self.model_path = model_path
        print("[AI] 모델 로드 완료 (준비 상태)")

    def generate_image(self, spectrogram_path, output_image_path="result.png"):
        """
        스펙트로그램 이미지를 받아 소리에 맞는 그림을 생성하는 함수
        """
        if not os.path.exists(spectrogram_path):
            print("[AI] 에러: 스펙트로그램 이미지가 없습니다.")
            return None
        
        # 2. [팀원 구현 과제] 스펙트로그램 이미지 전처리 및 모델 추론(Inference)
        # spec_img = Image.open(spectrogram_path)
        
        # 3. 임시 결과 생성 (뼈대 작동 확인용으로 원본을 복사하거나 더미 이미지 생성)
        # 지금은 뼈대이므로 입력받은 스펙트로그램을 그대로 결과인 척 저장합니다.
        dummy_result = Image.open(spectrogram_path)
        dummy_result.save(output_image_path)
        
        print(f"[AI] 이미지 생성 완료: {output_image_path}")
        return output_image_path

# 개별 테스트용 코드
if __name__ == "__main__":
    # ai = SoundToImageAI()
    # ai.generate_image("test_spec.png", "output.png")
    pass