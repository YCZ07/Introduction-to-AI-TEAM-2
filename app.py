# app.py
import gradio as gr
import librosa
import numpy as np
# 직접 작성하신 변환 함수 임포트
from audio_processor import convert_spectrogram_to_image
from ai_generator import ImageGenerationModel

# AI 모델 초기화
ai_model = ImageGenerationModel()

def process_audio(audio_data):
    if audio_data is None:
        return None, None
        
    # Gradio에서 넘겨주는 오디오 데이터 포맷: (sampling_rate, audio_array)
    sr, y = audio_data 
    
    # [임시 구현] 팀원이 할 역할: 원시 오디오를 dB 스케일의 스펙트로그램 2D 배열로 변환
    y = y.astype(np.float32)
    if len(y.shape) > 1: # 스테레오일 경우 모노로 변환
        y = np.mean(y, axis=1)
        
    stft_result = librosa.stft(y)
    spec_db = librosa.amplitude_to_db(np.abs(stft_result), ref=np.max)
    
    # 1. 작성하신 함수 실행 (배열 -> 이미지 변환)
    spectrogram_img = convert_spectrogram_to_image(spec_db, sr=sr)
    
    # 2. AI 모델 실행 (이미지 -> 최종 그림 생성)
    final_img = ai_model.generate_art(spectrogram_img)
    
    # 중간 확인을 위해 두 이미지를 모두 화면에 출력
    return spectrogram_img, final_img

# 웹 UI 구성
with gr.Blocks(title="음향 스펙트로그램 기반 이미지 생성기") as demo:
    gr.Markdown("## 🎵 소리 -> 스펙트로그램 -> 그림 변환 테스트")
    gr.Markdown("마이크로 녹음하거나 오디오 파일을 넣어보세요.")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone", "upload"], label="오디오 입력")
            submit_btn = gr.Button("변환 및 생성")
            
        with gr.Column():
            spec_output = gr.Image(label="중간 결과: 스펙트로그램", type="pil")
            art_output = gr.Image(label="최종 결과: AI 생성 그림", type="pil")
            
    submit_btn.click(
        fn=process_audio, 
        inputs=audio_input, 
        outputs=[spec_output, art_output]
    )

if __name__ == "__main__":
    demo.launch()