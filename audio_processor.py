# audio_processor.py (기존 파일에 아래 내용을 추가 및 수정하시면 됩니다)
import numpy as np
import matplotlib
# 웹 서버(Gradio) 환경에서 GUI 창이 뜨지 않도록 비대화형 백엔드 설정 (필수)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io
from PIL import Image

def convert_spectrogram_to_image(spec_db, sr=22050, hop_length=512, target_size=(256, 256)):
    """
    팀원이 처리한 스펙트로그램 데이터(dB)를 AI 모델 입력용 이미지 객체로 변환합니다.
    
    Args:
        spec_db (np.ndarray): 데시벨(dB) 스케일로 변환된 2D 스펙트로그램 행렬
        sr (int): 오디오 샘플링 레이트
        hop_length (int): STFT 연산 시 이동 간격
        target_size (tuple): AI 모델이 요구하는 최종 이미지 크기 (가로, 세로) omni
        
    Returns:
        PIL.Image: 변환된 스펙트로그램 이미지 객체
    """
    # 1. 그래프 플롯 초기화 (인공지능 모델 전처리용이므로 축, 여백을 모두 제거)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100) # figsize와 dpi로 기본 크기 제어
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    
    # 2. 스펙트로그램 시각화 (AI 모델의 특성에 따라 cmap 컬러맵을 변경할 수 있습니다)
    # 이미지 생성 AI가 컬러를 좋아하는지, 흑백(gray)을 좋아하는지에 따라 'viridis', 'magma', 'gray' 등 선택
    import librosa.display
    librosa.display.specshow(spec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', ax=ax, cmap='magma')
    
    # 3. 디스크에 파일로 저장하지 않고 메모리 버퍼(RAM)에 이미지를 이진 데이터로 저장
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    
    # 4. 메모리 버퍼의 데이터를 PIL 이미지 객체로 변환
    img = Image.open(buf)
    
    # 5. AI 모델이 요구하는 고정 규격 크기로 리사이징 (행렬 차원 맞추기)
    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # 6. 사용한 matplotlib 리소스 해제 (메모리 누수 방지)
    plt.close(fig)
    buf.close()
    
    print(f"[Audio Processor] 스펙트로그램 이미지 변환 완료: 크기 {img_resized.size}")
    return img_resized


# ==========================================
# 🧪 내 파트 독립 테스트를 위한 코드 (팀원에게 줄 때는 제외해도 됨)
# ==========================================
if __name__ == "__main__":
    print("독립 테스트를 시작합니다. 임의의 가상 스펙트로그램 행렬(데이터)을 생성합니다.")
    
    # 선형대수적인 접근: 주파수축 128, 시간축 256 개의 가상 dB 데이터 행렬 생성
    dummy_spec_db = np.random.uniform(-80, 0, (128, 256))
    
    # 함수 실행 테스트
    test_image = convert_spectrogram_to_image(dummy_spec_db, target_size=(256, 256))
    
    # 결과 확인을 위해 로컬에 임시 저장해보는 코드
    test_image.save("test_spectrogram_output.png")
    print("테스트 이미지 'test_spectrogram_output.png' 가 성공적으로 저장되었습니다.")