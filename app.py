import gradio as gr
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image, ImageFilter, ImageEnhance
import io, os, tempfile

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
    tempo      = float(np.atleast_1d(tempo)[0])   # 배열/스칼라 모두 안전하게 변환
    rms        = float(np.mean(librosa.feature.rms(y=y)))
    centroid   = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    rolloff    = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    zcr        = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    return dict(tempo=tempo, rms=rms,
                centroid=centroid, rolloff=rolloff, zcr=zcr)


# ══════════════════════════════════════════════════
#  3. 특성 → 아트워크 (PIL 기반, 외부 모델 없음)
# ══════════════════════════════════════════════════
STYLE_PALETTES = {
    "🖼️ 추상화":  ["#667eea", "#764ba2", "#f093fb", "#1a1a2e", "#43e97b"],
    "💧 수채화":  ["#a8edea", "#fed6e3", "#96fbc4", "#f9f586", "#b8c6db"],
    "🌸 파스텔":  ["#fbc2eb", "#a6c1ee", "#fddb92", "#d1fdff", "#f5f7fa"],
    "🎮 픽셀아트": ["#e60012", "#0067b9", "#f7d51d", "#000000", "#ffffff"],
    "🖌️ 유화":   ["#8B0000", "#DAA520", "#006400", "#00008B", "#2F4F4F"],
    "✏️ 스케치":  ["#1a1a1a", "#555555", "#aaaaaa", "#dddddd", "#ffffff"],
}

COLOR_TONES = {
    "🌈 선명하고 화려하게": (1.6, 1.4),
    "🖤 흑백으로":          (0.0, 1.0),
    "🍬 파스텔 톤으로":     (0.7, 0.8),
    "🌙 어둡고 신비롭게":   (1.2, 0.5),
    "☀️ 따뜻하고 밝게":    (0.9, 1.5),
}

def generate_artwork(spec_img: Image.Image,
                     features: dict,
                     style: str,
                     color_tone: str,
                     creativity: int) -> Image.Image:
    W, H = 512, 512
    rng  = np.random.default_rng(seed=int(features["tempo"] * 100))

    palette = STYLE_PALETTES.get(style, STYLE_PALETTES["🖼️ 추상화"])
    sat_mul, bri_mul = COLOR_TONES.get(color_tone, (1.0, 1.0))
    n_shapes = int(30 + creativity * 1.2)

    # 배경
    fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)
    fig.patch.set_facecolor(palette[-1])
    ax.set_facecolor(palette[-1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    tempo_norm    = min(features["tempo"] / 200, 1.0)
    energy_norm   = min(features["rms"] * 30, 1.0)
    bright_norm   = min(features["centroid"] / 8000, 1.0)
    chaos         = creativity / 100.0

    for _ in range(n_shapes):
        color = palette[rng.integers(len(palette))]
        alpha = rng.uniform(0.05 + energy_norm * 0.2, 0.35 + chaos * 0.3)
        x, y  = rng.uniform(0, 1), rng.uniform(0, 1)
        kind  = rng.integers(4)

        if kind == 0:   # 원
            r = rng.uniform(0.03, 0.15 + tempo_norm * 0.15)
            circle = plt.Circle((x, y), r, color=color, alpha=alpha)
            ax.add_patch(circle)
        elif kind == 1: # 타원
            w = rng.uniform(0.05, 0.3 + bright_norm * 0.2)
            h = rng.uniform(0.02, 0.15)
            angle = rng.uniform(0, 360)
            ellipse = matplotlib.patches.Ellipse((x, y), w, h, angle=angle,
                                                  color=color, alpha=alpha)
            ax.add_patch(ellipse)
        elif kind == 2: # 선
            x2 = x + rng.uniform(-0.4, 0.4) * chaos
            y2 = y + rng.uniform(-0.4, 0.4) * chaos
            lw = rng.uniform(0.5, 3 + energy_norm * 4)
            ax.plot([x, x2], [y, y2], color=color, alpha=alpha, lw=lw)
        else:           # 삼각형
            pts = rng.uniform(-0.12, 0.12, (3, 2)) + [x, y]
            tri = plt.Polygon(pts, color=color, alpha=alpha)
            ax.add_patch(tri)

    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).copy().resize((W, H))

    # 색감 보정
    if sat_mul != 0.0:
        img = ImageEnhance.Color(img).enhance(sat_mul)
    else:
        img = img.convert("L").convert("RGB")
    img = ImageEnhance.Brightness(img).enhance(bri_mul)

    # 스타일별 필터
    if style == "💧 수채화":
        img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    elif style == "✏️ 스케치":
        img = img.filter(ImageFilter.FIND_EDGES)
        img = ImageEnhance.Contrast(img).enhance(3.0)
    elif style == "🎮 픽셀아트":
        small = img.resize((64, 64), Image.NEAREST)
        img   = small.resize((W, H), Image.NEAREST)

    return img


# ══════════════════════════════════════════════════
#  4. 메인 파이프라인
# ══════════════════════════════════════════════════
def pipeline(audio_path, style, color_tone, creativity):
    if audio_path is None:
        return None, None, "⚠️ 음악 파일을 먼저 올려주세요!"

    try:
        spec_img = make_spectrogram(audio_path)
        features = extract_features(audio_path)
        art_img  = generate_artwork(spec_img, features, style, color_tone, int(creativity))

        info = (
            f"🎵 **분석 결과**\n"
            f"- 템포: {features['tempo']:.1f} BPM\n"
            f"- 에너지: {'높음 🔥' if features['rms'] > 0.05 else '낮음 🌙'}\n"
            f"- 주파수 중심: {features['centroid']:.0f} Hz\n"
            f"- 선택 스타일: {style}"
        )
        return spec_img, art_img, info

    except Exception as e:
        return None, None, f"❌ 오류 발생: {str(e)}"


# ══════════════════════════════════════════════════
#  5. UI
# ══════════════════════════════════════════════════
css = """
body { background: #f8f9fb !important; }
.gradio-container { max-width: 900px !important; margin: auto !important; font-family: -apple-system, sans-serif !important; }
.gr-button-primary {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    border: none !important; color: white !important;
    font-weight: 700 !important; font-size: 16px !important;
    height: 52px !important; border-radius: 12px !important;
}
.gr-button-primary:hover { opacity: 0.9 !important; }
footer { display: none !important; }
"""

STYLES     = list(STYLE_PALETTES.keys())
COLOR_OPTS = list(COLOR_TONES.keys())

with gr.Blocks(css=css, title="SonicVision — 소리를 그림으로") as demo:

    gr.Markdown("""
    # 🎵 SonicVision — 소리를 그림으로
    **음악 파일을 올리면 AI가 아름다운 그림을 만들어드려요.**  
    MP3, WAV, OGG, FLAC 파일을 지원합니다.
    """)

    with gr.Row():

        # ── 왼쪽: 입력 ──────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📥 1단계 — 음악 파일 올리기")
            audio_input = gr.Audio(
                label="여기에 파일을 끌어다 놓거나 클릭해서 올려주세요",
                type="filepath"
            )

            gr.Markdown("### 🎨 2단계 — 스타일 고르기")
            style_input = gr.Radio(
                choices=STYLES,
                value=STYLES[0],
                label="아트 스타일",
                info="원하는 그림 스타일을 골라주세요"
            )
            color_input = gr.Dropdown(
                choices=COLOR_OPTS,
                value=COLOR_OPTS[0],
                label="색감",
                info="그림의 전체적인 색 분위기예요"
            )
            creativity_input = gr.Slider(
                minimum=0, maximum=100, value=70, step=1,
                label="🎲 상상력 수준",
                info="숫자가 클수록 더 자유롭고 추상적인 그림이 나와요"
            )

            generate_btn = gr.Button("✨ 그림 만들기", variant="primary")

        # ── 오른쪽: 결과 ─────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📊 3단계 — 결과 확인")
            spec_output = gr.Image(
                label="소리 분석 이미지 (스펙트로그램)",
                interactive=False, height=240
            )
            art_output = gr.Image(
                label="🎨 AI가 만든 그림",
                interactive=False, height=240
            )
            info_output = gr.Markdown("")

    gr.Markdown("""
    ---
    <p style='text-align:center; color:#aaa; font-size:13px;'>
    SonicVision · 인공지능이 음악을 그림으로 변환합니다
    </p>
    """)

    generate_btn.click(
        fn=pipeline,
        inputs=[audio_input, style_input, color_input, creativity_input],
        outputs=[spec_output, art_output, info_output],
    )

if __name__ == "__main__":
    demo.launch(debug=True)