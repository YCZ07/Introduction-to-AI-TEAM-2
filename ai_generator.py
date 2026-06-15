# ai_generator.py
# ─────────────────────────────────────────────────────────────
# 실제 학습된 AI 모델(Stable Diffusion Turbo) 기반 이미지 생성기
#
# 스펙트로그램에서 뽑은 "소리 데이터"를 프롬프트(명령)로만 바꿔서
#         AI가 백지에서 진짜 미술 작품을 그림.
#
#   예) 시끄럽고 큰 소리 → 빨간색 격렬한 추상 표현주의 그림
#       조용한 소리      → 별이 빛나는 밤 같은 고요한 후기인상주의 그림
# ─────────────────────────────────────────────────────────────
import os
import random
import torch
from diffusers import AutoPipelineForText2Image


class ImageGenerationModel:
    def __init__(self, model_id: str = "stabilityai/sd-turbo"):
        print("[AI Model] 모델 로드 중... (첫 실행 시 약 2.5GB 다운로드, 수 분 소요)")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        # text2img 파이프라인 (백지에서 그림 생성)
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
        )
        self.pipe = self.pipe.to(self.device)
        self.pipe.enable_attention_slicing()

        if self.device == "cpu":
            torch.set_num_threads(os.cpu_count() or 4)

        print(f"[AI Model] 로드 완료! 사용 기기: {self.device}")

    # ── 소리 데이터(features) → 미술 작품 프롬프트 ───────────
    def _build_prompt(self, features: dict, style_hint: str, creativity: int) -> str:
        tempo    = float(features.get("tempo", 100))
        centroid = float(features.get("centroid", 2000))
        rms      = float(features.get("rms", 0.02))   # 소리 크기/에너지
        zcr      = float(features.get("zcr", 0.05))   # 노이즈성(거칠기)

        # 1) 가장 큰 줄기: 소리의 크기 → 작품의 전체 성격
        if rms > 0.06:        # 시끄럽고 큰 소리
            scene = ("a chaotic explosion of bold red and orange brushstrokes, "
                     "frantic scribbles and paint splatters, raw aggressive energy")
            mood = "abstract expressionism, action painting, turbulent, intense"
        elif rms > 0.025:     # 보통
            scene = "swirling dynamic forms bursting with movement and vivid color"
            mood = "expressive, energetic, vibrant"
        else:                 # 조용한 소리
            scene = "a calm dreamy landscape under a softly glowing swirling night sky"
            mood = "serene, tranquil, post-impressionist starry night, gentle, peaceful"

        # 2) 템포 → 붓질의 움직임
        if tempo > 120:
            motion = "fast swirling motion, sweeping energetic strokes"
        elif tempo < 80:
            motion = "slow flowing rhythm, smooth gentle strokes"
        else:
            motion = "balanced rhythmic flow"

        # 3) 주파수 중심(밝기) → 빛과 색
        if centroid > 3000:
            light = "bright luminous warm light, glowing highlights"
        elif centroid < 1500:
            light = "dark deep moody shadows, muted tones"
        else:
            light = "soft natural light"

        # 4) 노이즈성 → 질감
        texture = ("rough textured chaotic strokes" if zcr > 0.08
                   else "smooth harmonious blended colors")

        # 5) 상상력 슬라이더 → 구상 vs 추상
        if creativity > 65:
            abstraction = "highly abstract, non-figurative, experimental"
        elif creativity < 35:
            abstraction = "figurative, recognizable scenery, painterly"
        else:
            abstraction = "semi-abstract"

        return (f"{style_hint}, {scene}, {mood}, {motion}, {light}, {texture}, "
                f"{abstraction}, fine art masterpiece, painterly, no text")

    # ── 실제 생성 ────────────────────────────────────────────
    def generate_art(self,
                     features: dict = None,
                     style_hint: str = "oil painting",
                     creativity: int = 70):
        """
        소리 데이터(features)만으로 미술 작품을 생성.
        스펙트로그램 이미지는 더 이상 그림의 출발점으로 쓰지 않습니다.
        """
        features = features or {}
        prompt = self._build_prompt(features, style_hint, creativity)

        # 매번 다른 그림이 나오도록 무작위 시드 사용
        seed = random.randint(0, 2**31 - 1)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        print(f"[AI Model] 생성 중... prompt='{prompt[:60]}...'")
        result = self.pipe(
            prompt=prompt,
            num_inference_steps=1,   # 터보 모델: 1스텝 = 가장 빠름 (더 정교하게는 2~4)
            guidance_scale=0.0,      # 터보 모델 정석 설정
            height=512, width=512,
            generator=generator,
        ).images[0]

        return result


# 단독 테스트용
if __name__ == "__main__":
    model = ImageGenerationModel()
    # 시끄러운 곡 흉내
    loud = model.generate_art({"tempo": 140, "centroid": 3500, "rms": 0.09, "zcr": 0.12},
                              style_hint="oil painting", creativity=80)
    loud.save("test_loud.png")
    # 조용한 곡 흉내
    quiet = model.generate_art({"tempo": 70, "centroid": 1200, "rms": 0.01, "zcr": 0.03},
                               style_hint="oil painting", creativity=40)
    quiet.save("test_quiet.png")
    print("저장 완료: test_loud.png, test_quiet.png")