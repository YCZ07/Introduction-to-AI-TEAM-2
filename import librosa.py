import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import io

def convert_audio_to_spectrogram(audio_input, output_image_path="spectrogram.png"):
    """
    사용자가 업로드한 파일 경로, 파일 객체, 또는 녹음된 바이너리 데이터를 받아
    스펙트로그램 이미지(.png)로 변환하고 저장하는 함수.
    
    :param audio_input: 파일 경로(str) 또는 파일 객체/바이너리 버퍼(BytesIO 등)
    :param output_image_path: 저장할 스펙트로그램 이미지 파일 경로
    :return: 저장된 이미지 경로 (실패 시 None)
    """
    try:
        # 1. 오디오 데이터 로드
        # librosa.load는 파일 경로뿐만 아니라 파일 객체(BytesIO)도 지원합니다.
        # sr=None으로 설정하여 원본 오디오의 샘플링 레이트를 그대로 유지합니다.
        y, sr = librosa.load(audio_input, sr=None)
        
        # 2. STFT (Short-Time Fourier Transform) 계산
        # 시간 흐름에 따른 주파수 변화를 계산합니다.
        stft_result = librosa.stft(y)
        
        # 진폭(Amplitude)을 사람이 인지하기 좋은 데시벨(dB) 단위로 변환합니다.
        stft_db = librosa.amplitude_to_db(np.abs(stft_result), ref=np.max)
        
        # 3. Matplotlib를 이용해 스펙트로그램 시각화 및 저장
        # AI 모델 입력용(CNN 등)으로 사용할 것이기 때문에 축, 글자, 여백을 모두 제거합니다.
        fig, ax = plt.subplots(figsize=(5, 5))
        
        # specshow를 통해 스펙트로그램을 그립니다. 'magma'나 'viridis' 컬러맵을 주로 사용합니다.
        librosa.display.specshow(stft_db, sr=sr, ax=ax, cmap='magma')
        
        # 테두리 및 축 제거 (순수 이미지 데이터만 남기기 위함)
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        # 이미지 파일로 저장 (해상도를 높이려면 dpi를 조절할 수 있습니다)
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close(fig) # 메모리 해제를 위해 figure를 닫아줍니다.
        
        print(f"[Audio] 스펙트로그램 저장 완료: {output_image_path}")
        return output_image_path

    except Exception as e:
        print(f"[Audio] 에러 발생: {e}")
        return None

# 오디오 담당자가 혼자서 로컬 환경에서 테스트해볼 수 있는 구문
if __name__ == "__main__":
    print("오디오 처리 모듈 단독 테스트 시작")
    
    # 확장자가 wav인지 mp3인지 모르겠다면 아래 둘 중 맞는 것으로 주석(#)을 풀어서 쓰세요!
    # 1. 파일이 MP3 파일일 때:
    test_audio = "테스트1.mp3" 
    
    # 2. 파일이 WAV 파일일 때 (위의 mp3가 아니라면 이 줄 맨 앞의 #을 지우고 위를 주석처리 하세요):
    # test_audio = "테스트1.wav" 
    
    output_image = "test_spectrogram.png"
    
    print(f"[{test_audio}] 파일을 스펙트로그램으로 변환합니다...")
    result = convert_audio_to_spectrogram(test_audio, output_image)
    
    if result:
        print(f"✨ 성공! 결과 이미지가 저장되었습니다: {result}")
    else:
        print("❌ 실패! 파일 이름이나 확장자를 다시 확인해보세요.")