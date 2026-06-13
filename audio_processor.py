# audio_processor.py
import librosa
import librosa.display
import numpy as np
import matplotlib
matplotlib.use('Agg') # GUI 에러 방지
import matplotlib.pyplot as plt
import io
from PIL import Image

def process_audio_to_image(sr, y, target_size=(256, 256)):
    """
    Gradio에서 넘겨받은 오디오 배열을 STFT 행렬 연산을 거쳐
    최종 AI 모델 입력용 스펙트로그램 이미지로 한 번에 변환합니다.
    """
    # 1. 스테레오일 경우 모노로 변환 및 타입 지정
    y = y.astype(np.float32)
    if len(y.shape) > 1: 
        y = np.mean(y, axis=1)

    # 2. (팀원 로직) STFT 계산 및 dB 스케일 변환
    stft_result = librosa.stft(y)
    spec_db = librosa.amplitude_to_db(np.abs(stft_result), ref=np.max)

    # 3. (본인 로직) 행렬 데이터를 스펙트로그램 시각화 후 메모리 버퍼에 저장
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')

    librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='log', ax=ax, cmap='magma')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)

    # 4. PIL 이미지 객체로 변환 및 리사이징
    img = Image.open(buf)
    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)

    plt.close(fig)
    buf.close()

    print(f"[Audio] 스펙트로그램 통합 변환 완료: 크기 {img_resized.size}")
    return img_resized