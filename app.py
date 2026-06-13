# app.py
import gradio as gr
from audio_processor import process_audio_to_image
from ai_generator import ImageGenerationModel

# 서버 시작 시 AI 모델 초기화
ai_model = ImageGenerationModel()

def process_pipeline(audio_data):
    """오디오 입력부터 이미지 출력까지의 전체 흐름을 제어합니다."""
    if audio_data is None:
        return None, None
        
    sr, y = audio_data 
    
    # 1. 오디오 -> 스펙트로그램 이미지 변환
    spectrogram_img = process_audio_to_image(sr, y)
    
    # 2. 스펙트로그램 이미지 -> 최종 그림 생성
    final_img = ai_model.generate_art(spectrogram_img)
    
    return spectrogram_img, final_img

# 웹 UI 구성
with gr.Blocks(title="음향 스펙트로그램 기반 이미지 생성기") as demo:
    gr.Markdown("## 🎵 소리 -> 스펙트로그램 -> 그림 변환 테스트")
    gr.Markdown("마이크로 직접 소리를 녹음하거나 오디오 파일을 업로드해보세요.")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone", "upload"], label="오디오 입력")
            submit_btn = gr.Button("변환 및 생성")
            
        with gr.Column():
            spec_output = gr.Image(label="1단계: 분석된 스펙트로그램", type="pil")
            art_output = gr.Image(label="2단계: AI 최종 생성 그림", type="pil")
            
    submit_btn.click(
        fn=process_pipeline, 
        inputs=audio_input, 
        outputs=[spec_output, art_output]
    )

if __name__ == "__main__":
    demo.launch()