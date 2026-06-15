import gradio as gr
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import io, os

# ══════════════════════════════════════════════════
#  1. 스펙트로그램 생성
# ══════════════════════════════════════════════════
def make_spectrogram(audio_path: str) -> Image.Image:
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel",
                             ax=ax, cmap="magma", fmax=8000)
    ax.set_xlabel("시간 (초)", color="white", fontsize=9)
    ax.set_ylabel("주파수 (Hz)", color="white", fontsize=9)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ══════════════════════════════════════════════════
#  2. 오디오 특성 추출
# ══════════════════════════════════════════════════
def extract_features(audio_path: str) -> dict:
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    tempo, _   = librosa.beat.beat_track(y=y, sr=sr)
    tempo      = float(np.atleast_1d(tempo)[0])
    rms        = float(np.mean(librosa.feature.rms(y=y)))
    centroid   = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    rolloff    = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    zcr        = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    return dict(tempo=tempo, rms=rms,
                centroid=centroid, rolloff=rolloff, zcr=zcr)


# ══════════════════════════════════════════════════
#  3. 실제 AI 모델 연결 (SD-Turbo)
# ══════════════════════════════════════════════════
_ai_model = None

def get_ai_model():
    global _ai_model
    if _ai_model is None:
        from ai_generator import ImageGenerationModel
        _ai_model = ImageGenerationModel()
    return _ai_model

STYLE_PROMPTS = {
    "🖼️ 추상화":   "abstract painting",
    "💧 수채화":   "watercolor painting, soft wet brush strokes",
    "🌸 파스텔":   "pastel art, soft pastel colors",
    "🎮 픽셀아트": "pixel art, 8-bit retro game style",
    "🖌️ 유화":    "oil painting, thick textured brush strokes",
    "✏️ 스케치":   "pencil sketch, black and white line drawing",
}

COLOR_TONE_PROMPTS = {
    "🌈 선명하고 화려하게": "vivid saturated colors",
    "🖤 흑백으로":          "black and white, monochrome",
    "🍬 파스텔 톤으로":     "pastel tones, light colors",
    "🌙 어둡고 신비롭게":   "dark mysterious atmosphere",
    "☀️ 따뜻하고 밝게":    "warm bright lighting",
}


# ══════════════════════════════════════════════════
#  4. 입력 소스 처리 (녹음 / 파일 선택)
# ══════════════════════════════════════════════════
def on_upload(path):
    """파일 버튼으로 선택 → 업로드 경로 저장, 라벨 표시"""
    if not path:
        return None, ""
    name = os.path.basename(path)
    return path, f"선택된 파일: **{name}**"

def on_record(_mic):
    """새로 녹음하면 업로드 선택은 해제 → 녹음을 사용"""
    return None, ""


# ══════════════════════════════════════════════════
#  5. 메인 파이프라인
# ══════════════════════════════════════════════════
def pipeline(mic_path, uploaded_path, style, color_tone, creativity):
    # 업로드한 파일이 있으면 우선, 없으면 녹음 사용
    audio_path = uploaded_path or mic_path
    if audio_path is None:
        return None, None, "음악을 녹음하거나 파일을 선택해주세요."

    try:
        spec_img = make_spectrogram(audio_path)
        features = extract_features(audio_path)

        style_hint = STYLE_PROMPTS.get(style, "abstract art")
        tone_hint  = COLOR_TONE_PROMPTS.get(color_tone, "")
        full_hint  = f"{style_hint}, {tone_hint}".strip(", ")

        model   = get_ai_model()
        art_img = model.generate_art(features,
                                     style_hint=full_hint,
                                     creativity=int(creativity))

        energy_label = "높음" if features["rms"] > 0.05 else "낮음"
        info = (
            f"**분석 결과 (AI 생성)** ｜ "
            f"템포 {features['tempo']:.0f} BPM ｜ "
            f"에너지 {energy_label} ｜ "
            f"주파수 중심 {features['centroid']:.0f} Hz ｜ "
            f"스타일 {style}"
        )
        return spec_img, art_img, info

    except Exception as e:
        return None, None, f"오류 발생: {str(e)}"


# ══════════════════════════════════════════════════
#  6. UI
# ══════════════════════════════════════════════════
ICON_MIC = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="2" width="6" height="11" rx="3"/><path d="M5 10a7 7 0 0 0 14 0"/><line x1="12" y1="17" x2="12" y2="21"/><line x1="8" y1="21" x2="16" y2="21"/></svg>'
ICON_BRUSH = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 21c2.8 0 4.5-1.8 4.5-4.5"/><path d="M14.7 4.6l4.7 4.7L11 17.5l-4.5.9.9-4.5z"/></svg>'
ICON_WAVE = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="13" x2="4" y2="18"/><line x1="9" y1="7" x2="9" y2="18"/><line x1="14" y1="4" x2="14" y2="18"/><line x1="19" y1="10" x2="19" y2="18"/></svg>'
ICON_FRAME = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.6"/><path d="M21 15l-5-5L5 21"/></svg>'

def step_header(icon_svg: str, text: str) -> str:
    return f'<div class="step-head"><span class="ico">{icon_svg}</span><span class="t">{text}</span></div>'

LOGO_HTML = """
<div class="logo-row">
  <svg class="logo-note" width="38" height="38" viewBox="0 0 36 36" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="svNote" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stop-color="#667eea"/>
        <stop offset="0.55" stop-color="#764ba2"/>
        <stop offset="1" stop-color="#f093fb"/>
      </linearGradient>
    </defs>
    <g fill="url(#svNote)">
      <path d="M12.5 10 L25.9 6.5 L25.9 10.6 L12.5 14.1 Z"/>
      <rect x="12.5" y="10"  width="2.6" height="17.6" rx="1.1"/>
      <rect x="23.3" y="6.5" width="2.6" height="16.6" rx="1.1"/>
      <ellipse cx="9.4"  cy="27.6" rx="4.7" ry="3.5"/>
      <ellipse cx="20.2" cy="23.1" rx="4.7" ry="3.5"/>
    </g>
  </svg>
  <div class="logo-text">
    <div class="logo-title">SonicVision</div>
    <div class="logo-sub">소리를 그림으로 — AI가 음악을 아름다운 그림으로 바꿔드려요</div>
  </div>
</div>
"""

css = """
html, body, gradio-app, .gradio-container, .main, .app, .contain { background: transparent; }
body, gradio-app {
    background: linear-gradient(135deg, #eef1ff 0%, #f7eefc 45%, #ffeef7 100%) !important;
    min-height: 100vh !important;
}
.gradio-container {
    max-width: 1700px !important;
    width: 96% !important;
    margin: auto !important;
    padding: 0px 16px 2px 16px !important;
    background: transparent !important;
    font-family: -apple-system, 'Segoe UI', sans-serif !important;
}

/* 로고 */
.logo-row { display:flex; align-items:center; gap:10px; padding:1px 4px; }
.logo-note { flex:0 0 auto; width:34px; height:34px; filter: drop-shadow(0 2px 6px rgba(118,75,162,0.26)); }
.logo-title {
    font-size: 22px; font-weight: 800; line-height: 1.0; letter-spacing: -0.5px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 55%, #f093fb 100%);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; color: transparent;
    display: inline-block;
}
.logo-sub { font-size: 11.5px; color: #6b6b80; margin-top: 1px; }

/* 단계 제목 */
.step-head { display:flex; align-items:center; gap:8px; margin: 1px 0 7px 0; }
.step-head .ico { display:inline-flex; }
.step-head svg { width:18px; height:18px; color:#7c4dbb; }
.step-head .t { font-size:15px; font-weight:700; color:#2b2540; }

/* 카드형 패널 (여유롭게) */
.panel {
    background: rgba(255,255,255,0.66) !important;
    border: 1px solid rgba(118,75,162,0.14) !important;
    border-radius: 16px !important;
    padding: 14px 18px !important;
    box-shadow: 0 4px 18px rgba(118,75,162,0.08) !important;
}

/* 기본(생성) 버튼 */
.gr-button-primary, button.primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 55%, #f093fb 100%) !important;
    border: none !important; color: white !important;
    font-weight: 700 !important; font-size: 15px !important;
    height: 44px !important; border-radius: 12px !important;
}
.gr-button-primary:hover, button.primary:hover { opacity: 0.92 !important; }

/* 파일 선택 버튼 */
.upload-btn button, button.upload-btn {
    background: rgba(255,255,255,0.75) !important;
    border: 1.5px solid rgba(118,75,162,0.35) !important;
    color: #6b4ba0 !important; font-weight: 600 !important;
    height: 40px !important; border-radius: 10px !important;
}
.file-label p { font-size: 12.5px !important; color: #6b6b80 !important; margin: 2px 0 0 2px !important; }

/* 간격 */
.gradio-container .form { gap: 9px !important; }
.gradio-container .prose p { margin: 1px 0 !important; }
.gradio-container .gap { gap: 12px !important; }

/* 🎤 마이크(녹음) 위젯
   [수정] 기존 max-height:150px + overflow:hidden 이 녹음 파형/재생 컨트롤을
   잘라버려서 "이미지가 깨지는" 현상이 있었음 → 잘리지 않게 풀어주고,
   파형 캔버스가 칸 밖으로 넘치지 않도록 너비만 100%로 제한. */
.audio-box { overflow: visible !important; }
.audio-box, .audio-box .wrap, .audio-box > div { background: rgba(255,255,255,0.45) !important; }
.audio-box canvas, .audio-box .waveform-container, .audio-box .component-wrap {
    max-width: 100% !important;
}

/* 빈 이미지/여백의 흰색 → 연보라 색감 */
.big-img, .big-img > div, .big-img .image-frame, .big-img .image-container,
.big-img .empty, .big-img .wrap, .big-img [data-testid="image"] {
    background: linear-gradient(135deg, #f3f0ff 0%, #fbf1fb 100%) !important;
    border-radius: 12px !important;
}
.big-img .empty svg, .big-img .empty { color: #c3b3e6 !important; }
.big-img img { background: transparent !important; }

footer { display: none !important; }
"""

with gr.Blocks(css=css, title="SonicVision — 소리를 그림으로") as demo:

    gr.HTML(LOGO_HTML)

    uploaded_state = gr.State(None)   # '파일에서 선택' 버튼으로 고른 경로 저장

    # ════════ 한 줄에 3칸 : [녹음] [설정] [스펙트로그램+완성작품] ════════
    #   1 : 1  : 2 구조 — 셋째 칸만 이미지 2개를 세로로 쌓음.
    #   → 가로로 넓고 세로는 짧음. 녹음으로 1칸이 길어져도 셋째 칸이 더 높아서
    #     전체 높이(=가장 높은 칸)가 흔들리지 않음.
    with gr.Row(equal_height=True):

        # ── 1칸 : 음악 입력(녹음) ───────────────────
        with gr.Column(scale=4, elem_classes="panel"):
            gr.HTML(step_header(ICON_MIC, "1단계 — 음악 입력하기"))

            # 마이크 녹음 + 파일 드래그앤드롭(업로드) 둘 다 지원
            audio_input = gr.Audio(
                sources=["upload", "microphone"],   # ← 드래그앤드롭(업로드) 포함
                type="filepath",
                label="🎤 녹음하거나, 음악 파일을 여기로 끌어다 놓으세요",
                elem_classes="audio-box"
            )
            # 클릭해서 파일 탐색기로 직접 고르고 싶을 때
            upload_btn = gr.UploadButton(
                "또는 파일에서 선택",
                file_types=[".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"],
                type="filepath",
                elem_classes="upload-btn"
            )
            file_label = gr.Markdown("", elem_classes="file-label")

        # ── 2칸 : 스타일 & 색감 설정 ────────────────
        with gr.Column(scale=4, elem_classes="panel"):
            gr.HTML(step_header(ICON_BRUSH, "2단계 — 스타일 & 색감 고르기"))
            style_input = gr.Radio(
                choices=list(STYLE_PROMPTS.keys()),
                value=list(STYLE_PROMPTS.keys())[0],
                label="아트 스타일",
            )
            with gr.Row():
                with gr.Column(min_width=140):
                    color_input = gr.Dropdown(
                        choices=list(COLOR_TONE_PROMPTS.keys()),
                        value=list(COLOR_TONE_PROMPTS.keys())[0],
                        label="색감",
                    )
                with gr.Column(min_width=140):
                    creativity_input = gr.Slider(
                        minimum=0, maximum=100, value=70, step=1,
                        label="상상력 수준",
                        info="숫자가 클수록 더 자유롭고 추상적인 그림이 나와요",
                    )
            generate_btn = gr.Button("그림 만들기", variant="primary")

        # ── 3칸 : 스펙트로그램(위) + 완성 작품(아래) ──
        with gr.Column(scale=5, elem_classes="panel"):
            # (위) 소리 분석 — 스펙트로그램
            gr.HTML(step_header(ICON_WAVE, "3단계 — 소리 분석"))
            spec_output = gr.Image(
                label="소리 분석 이미지 (스펙트로그램)",
                interactive=False, height=170, elem_classes="big-img"   # 높이 조절: 이 숫자만 바꾸세요
            )
            # (아래) 완성 작품 — 생성된 이미지
            gr.HTML(step_header(ICON_FRAME, "완성 작품"))
            art_output = gr.Image(
                label="AI가 만든 그림",
                interactive=False, height=240, elem_classes="big-img"   # 높이 조절: 이 숫자만 바꾸세요
            )
            info_output = gr.Markdown("")

    # ── 이벤트 연결 ──
    # 파일 버튼으로 선택
    upload_btn.upload(on_upload, inputs=upload_btn, outputs=[uploaded_state, file_label])
    # 오디오 칸에 파일을 드래그앤드롭(업로드)하면 그게 최신 입력이 되도록 버튼 선택은 해제
    audio_input.upload(on_record, inputs=audio_input, outputs=[uploaded_state, file_label])
    # 새로 녹음해도 버튼 선택 해제
    audio_input.stop_recording(on_record, inputs=audio_input, outputs=[uploaded_state, file_label])

    generate_btn.click(
        fn=pipeline,
        inputs=[audio_input, uploaded_state, style_input, color_input, creativity_input],
        outputs=[spec_output, art_output, info_output],
    )

if __name__ == "__main__":
    demo.launch(debug=True)