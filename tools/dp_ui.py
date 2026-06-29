#!/usr/bin/env python3
import os
import shlex
import signal
import subprocess
import time
import tomllib
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_DIR / "configs" / "generated"
RUN_ROOT = REPO_DIR / "training_runs"
LOG_ROOT = REPO_DIR / "ui_logs"
TRAIN_PID_FILE = LOG_ROOT / "current_train.pid"

MODEL_LABELS = {
    # Flux family
    "Flux Dev": "flux",
    "Flux Kontext": "flux_kontext",
    "Flux 2 Dev": "flux2_dev",
    "Flux 2 Klein 4B": "flux2_klein4b",
    "Flux 2 Klein 9B": "flux2_klein9b",
    "Chroma": "chroma",
    # Qwen / Z-Image family
    "Qwen-Image": "qwen_image",
    "Qwen-Image-Edit": "qwen_image_edit",
    "Z-Image": "z_image",
    # Hunyuan family
    "HunyuanImage-2.1": "hunyuan_image",
    "HunyuanVideo": "hunyuan_video",
    "HunyuanVideo-1.5": "hunyuan_video_15",
    "HiDream": "hidream",
    # LTX family
    "LTX-Video": "ltx_video",
    "LTX 2.3": "ltx2",
    # Wan family
    "Wan2.1": "wan21",
    "Wan2.2 (Low Noise)": "wan22_low",
    "Wan2.2 (High Noise)": "wan22_high",
    # Others
    "Lumina Image 2.0": "lumina_2",
    "Cosmos": "cosmos",
    "Cosmos-Predict2": "cosmos_predict2",
    "OmniGen2": "omnigen2",
    "Ideogram4": "ideogram4",
    "Ernie-Image": "ernie_image",
    "Anima": "anima",
    "Krea 2": "krea2",
    "AuraFlow": "auraflow",
    "SD3": "sd3",
    "SDXL": "sdxl",
}

# UI metadata per model key:
# main_label, vae_label (None=hidden), te_label (None=hidden), te2_label (None=hidden)
# adapter_label (None=hidden), show_shift (numeric), show_flux_shift (checkbox)
# show_max_seq, show_hidream_4bit, show_min_max_t, show_llm_lr, show_sdxl_lr
# shift_default, min_t_default, max_t_default, max_seq_default, notes
MODEL_UI = {
    "flux": dict(
        main_label="Diffusers folder (FLUX.1-dev or Schnell)",
        vae_label=None, te_label=None, te2_label=None,
        adapter_label="Transformer .safetensors (optional override)",
        show_shift=False, show_flux_shift=True, show_max_seq=False,
        show_hidream_4bit=False, show_min_max_t=False, show_llm_lr=False, show_sdxl_lr=False,
        shift_default=1, min_t_default=0.0, max_t_default=1.0, max_seq_default=256,
        notes="VAE and text encoders are loaded from the Diffusers folder. LoRA saved in Diffusers format.",
    ),
    "flux_kontext": dict(
        main_label="Diffusers folder (FLUX.1-dev)",
        vae_label=None, te_label=None, te2_label=None,
        adapter_label="Kontext transformer .safetensors (flux1-kontext-dev)",
        show_shift=False, show_flux_shift=False, show_max_seq=False,
        show_hidream_4bit=False, show_min_max_t=False, show_llm_lr=False, show_sdxl_lr=False,
        shift_default=1, min_t_default=0.0, max_t_default=1.0, max_seq_default=256,
        notes="Compatible with Flux Dev weights. Set dataset like Flux Kontext example. LoRA saved in Diffusers format.",
    ),
    "flux2_dev": dict(
        main_label="diffusion_model .safetensors (flux2-dev)",
        vae_label="VAE .safetensors (flux2-vae)",
        te_label="Text encoder .safetensors (mistral_3_small_flux2_fp8)",
        te2_label=None, adapter_label=None,
        show_shift=True, show_flux_shift=False, show_max_seq=False,
        show_hidream_4bit=False, show_min_max_t=False, show_llm_lr=False, show_sdxl_lr=False,
        shift_default=3, min_t_default=0.0, max_t_default=1.0, max_seq_default=256,
        notes="shift=3 recommended. Use ComfyUI-compatible weights. LoRA saved in ComfyUI format.",
    ),
    "flux2_klein4b": dict(
        main_label="diffusion_model .safetensors (flux-2-klein-base-4b)",
        vae_label="VAE .safetensors (flux2-vae)",
        te_label="Text encoder .safetensors (qwen_3_4b)",
        te2_label=None, adapter_label=None,
        show_shift=True, show_flux_shift=False, show_max_seq=False,
        show_hidream_4bit=False, show_min_max_t=False, show_llm_lr=False, show_sdxl_lr=False,
        shift_default=3, min_t_default=0.0, max_t_default=1.0, max_seq_default=256,
        notes="Use the BASE model (not distilled). shift=3 default. LoRA saved in ComfyUI format.",
    ),
    "flux2_klein9b": dict(
        main_label="diffusion_model .safetensors (flux-2-klein-base-9b)",
        vae_label="VAE .safetensors (flux2-vae)",
        te_label="Text encoder .safetensors (qwen_3_8b)",
        te2_label=None, adapter_label=None,
        show_shift=True, show_flux_shift=False, show_max_seq=False,
        show_hidream_4bit=False, show_min_max_t=False, show_llm_lr=False, show_sdxl_lr=False,
        shift_default=3, min_t_default=0.0, max_t_default=1.0, max_seq_default=256,
        notes="Use the BASE model (not distilled). shift=3 default. LoRA saved in ComfyUI format.",
    ),
    "chroma": dict(
        main_label="Diffusers folder (FLUX.1-dev or Schnell)",
        vae_label=None, te_label=None, te2_label=None,
        adapter_label="Chroma transformer .safetensors (required, e.g. chroma-unlocked-v10)",
        show_shift=False, show_flux_shift=True, show_max_seq=False,
        show_hidream_4bit=False, show_min_max_t=False, show_llm_lr=False, show_sdxl_lr=False,
        shift_default=1, min_t_default=0.0, max_t_default=1.0, max_seq_default=256,
        notes="Diffusers folder provides VAE and text encoder. Chroma transformer is the single file. LoRA in ComfyUI format.",
    ),
    "qwen_image": dict(
        main_label="Diffusers folder (Qwen-Image)  — or transformer .safetensors for individual files",
        vae_label="VAE path (Diffusers VAE, only needed with individual files)",
        te_label="Text encoder .safetensors (qwen_2.5_vl_7b, only needed with individual files)",
        te2_label=None, adapter_label=None,
        show_shift=False, show_flux_shift=False, show_max_seq=False,
        show_hidream_4bit=False, show_min_max_t=False, show_llm_lr=False, show_sdxl_lr=False,
        shift_default=1, min_t_default=0.0, max_t_default=1.0, max_seq_default=256,
        notes="Can use Diffusers folder only (leave VAE/TE blank), or specify individual ComfyUI files. LoRA in ComfyUI format.",
    ),
    "qwen_image_edit": dict(
        main_label="Diffusers folder (Qwen-Image or Qwen-Image-Edit)",
        vae_label=None, te_label=None, te2_label=None,
        adapter_label="Qwen-Image-Edit transformer .safetensors (if using Qwen-Image Diffusers folder)",
        show_shift=False, show_flux_shift=False, show_max_seq=False,
        show_hidream_4bit=False, show_min_max_t=False, show_llm_lr=False, show_sdxl_lr=False,
        shift_default=1, min_t_default=0.0, max_t_default=1.0, max_seq_default=256,
        notes="Configure dataset like Flux Kontext example. Reference images must match target aspect ratio. LoRA in ComfyUI format.",
    ),
    "z_image": dict(
        main_label="diffusion_model .safetensors (z_image_turbo_bf16)",
        vae_label="VAE .safetensors (flux_vae)",
        te_label="Text encoder .safetensors (qwen_3_4b)",
        te2_label=None,
        adapter_label="Turbo training adapter .safetensors (required for Z-Image-Turbo)",
        show_shift=False, show_flux_shift=False, show_max_seq=False,
        show_hidream_4bit=False, show_min_max_t=False, show_llm_lr=False, show_sdxl_lr=False,
        shift_default=1, min_t_default=0.0, max_t_default=1.0, max_seq_default=256,
        notes="All ComfyUI format files from Comfy-Org/z_image_turbo. For Turbo, include the training adapter. LoRA in ComfyUI format.",
    ),
    "hunyuan_image": dict(
        main_label="transformer .safetensors (hunyuanimage2.1)",
        vae_label="VAE .safetensors (hunyuan_image_2.1_vae_fp16)",
        te_label="Text encoder .safetensors (qwen_2.5_vl_7b)",
        te2_label="byt5 .safetensors (byt5_small_glyphxl_fp16)",
        adapter_label=None,
        show_shift=False, show_flux_shift=False, show_max_seq=False,
        show_hidream_4bit=False, show_min_max_t=False, show_llm_lr=False, show_sdxl_lr=False,
        shift_default=1, min_t_default=0.0, max_t_default=1.0, max_seq_default=256,
        notes="Note: 1024 res here ≈ 512 for Flux/Qwen. All ComfyUI format files. LoRA in ComfyUI format.",
    ),
    "hunyuan_video": dict(
        main_label="transformer .safetensors (hunyuan_video_720_cfgdistill_fp8)",
        vae_label="VAE .safetensors (hunyuan_video_vae_bf16)",
        te_label="LLM folder (llava-llama-3-8b-text-encoder-tokenizer)",
        te2_label="CLIP folder (clip-vit-large-patch14)",
        adapter_label=None,
        show_shift=False, show_flux_shift=False, show_max_seq=False,
        show_hidream_4bit=False, show_min_max_t=False, show_llm_lr=False, show_sdxl_lr=False,
        shift_default=1, min_t_default=0.0, max_t_default=1.0, max_seq_default=256,
        notes="Load from ComfyUI files. LoRA saved in Diffusers-style format, compatible with ComfyUI.",
    ),
    "hunyuan_video_15": dict(
        main_label="diffusion_model .safetensors (hunyuanvideo1.5_480p_t2v_fp16)",
        vae_label="VAE .safetensors (hunyuanvideo15_vae_fp16)",
        te_label="Text encoder 1 .safetensors (qwen_2.5_vl_7b)",
        te2_label="Text encoder 2 .safetensors (byt5_small_glyphxl_fp16)",
        adapter_label=None,
        show_shift=False, show_flux_shift=False, show_max_seq=False,
        show_hidream_4bit=False, show_min_max_t=False, show_llm_lr=False, show_sdxl_lr=False,
        shift_default=1, min_t_default=0.0, max_t_default=1.0, max_seq_default=256,
        notes="All ComfyUI format files. LoRA saved in ComfyUI format.",
    ),
    "hidream": dict(
        main_label="Diffusers folder (HiDream-I1-Full)",
        vae_label=None,
        te_label="Llama3 path (Meta-Llama-3.1-8B-Instruct folder)",
        te2_label=None, adapter_label=None,
        show_shift=False, show_flux_shift=True, show_max_seq=True,
        show_hidream_4bit=True, show_min_max_t=False, show_llm_lr=False, show_sdxl_lr=False,
        shift_default=1, min_t_default=0.0, max_t_default=1.0, max_seq_default=128,
        notes="Only Full version supported (not Dev/Fast). 4-bit Llama3 saves VRAM with no measurable quality loss. LoRA in ComfyUI format.",
    ),
    "ltx_video": dict(
        main_label="Diffusers folder (LTX-Video)",
        vae_label=None, te_label=None, te2_label=None,
        adapter_label="single_file_path .safetensors (optional, for newer LTX versions like v0.9.1)",
        show_shift=False, show_flux_shift=False, show_max_seq=False,
        show_hidream_4bit=False, show_min_max_t=False, show_llm_lr=False, show_sdxl_lr=False,
        shift_default=1, min_t_default=0.0, max_t_default=1.0, max_seq_default=256,
        notes="Diffusers folder still needed for text encoder. LoRA saved in ComfyUI format.",
    ),
    "ltx2": dict(
        main_label="diffusion_model .safetensors (ltx-2.3-22b-dev)",
        vae_label=None,
        te_label="text_encoder .safetensors (gemma_3_12B_it_fp4_mixed)",
        te2_label=None, adapter_label=None,
        show_shift=True, show_flux_shift=False, show_max_seq=False,
        show_hidream_4bit=False, show_min_max_t=False, show_llm_lr=False, show_sdxl_lr=False,
        shift_default=1, min_t_default=0.0, max_t_default=1.0, max_seq_default=256,
        notes="Only LTX 2.3 supported. Max blocks_to_swap=46. 24GB tight — use low resolution and rank. LoRA in ComfyUI format.",
    ),
    "wan21": dict(
        main_label="ckpt_path (Wan2.1-T2V-1.3B or 14B folder)",
        vae_label="Optional: transformer .safetensors override (ComfyUI repackaged)",
        te_label="Optional: LLM .safetensors override (umt5-xxl-enc-bf16)",
        te2_label=None, adapter_label=None,
        show_shift=False, show_flux_shift=False, show_max_seq=False,
        show_hidream_4bit=False, show_min_max_t=False, show_llm_lr=False, show_sdxl_lr=False,
        shift_default=1, min_t_default=0.0, max_t_default=1.0, max_seq_default=256,
        notes="Set ckpt_path to HuggingFace checkpoint folder. VAE/TE fields are optional ComfyUI overrides. LoRA in ComfyUI format.",
    ),
    "wan22_low": dict(
        main_label="ckpt_path (Wan2.2-T2V-A14B folder)",
        vae_label="Optional: transformer .safetensors (wan2.2_t2v_low_noise_14B_fp16)",
        te_label="Optional: LLM .safetensors (umt5_xxl_fp16)",
        te2_label=None, adapter_label=None,
        show_shift=False, show_flux_shift=False, show_max_seq=False,
        show_hidream_4bit=False, show_min_max_t=True, show_llm_lr=False, show_sdxl_lr=False,
        shift_default=1, min_t_default=0.0, max_t_default=0.875, max_seq_default=256,
        notes="Low noise model handles timesteps 0→0.875. Use min_t=0, max_t=0.875 for T2V. LoRA in ComfyUI format.",
    ),
    "wan22_high": dict(
        main_label="ckpt_path (Wan2.2-T2V-A14B folder)",
        vae_label="Optional: transformer .safetensors (wan2.2_t2v_high_noise_14B_fp16)",
        te_label="Optional: LLM .safetensors (umt5_xxl_fp16)",
        te2_label=None, adapter_label=None,
        show_shift=False, show_flux_shift=False, show_max_seq=False,
        show_hidream_4bit=False, show_min_max_t=True, show_llm_lr=False, show_sdxl_lr=False,
        shift_default=1, min_t_default=0.875, max_t_default=1.0, max_seq_default=256,
        notes="High noise model handles timesteps 0.875→1.0. Use min_t=0.875, max_t=1.0 for T2V. LoRA in ComfyUI format.",
    ),
    "lumina_2": dict(
        main_label="transformer .safetensors (lumina_2_model_bf16)",
        vae_label="VAE .safetensors (flux_vae)",
        te_label="LLM .safetensors (gemma_2_2b_fp16)",
        te2_label=None, adapter_label=None,
        show_shift=False, show_flux_shift=False, show_max_seq=False,
        show_hidream_4bit=False, show_min_max_t=False, show_llm_lr=False, show_sdxl_lr=False,
        shift_default=1, min_t_default=0.0, max_t_default=1.0, max_seq_default=256,
        notes="lumina_shift=true added automatically. Supports FFT at 1024px on a single 24GB GPU. LoRA in ComfyUI format.",
    ),
    "cosmos": dict(
        main_label="transformer_path .pt (cosmos-1.0-diffusion-7b-text2world)",
        vae_label="VAE .safetensors (cosmos_cv8x8x8_1.0)",
        te_label="text_encoder .safetensors (oldt5_xxl_fp16)",
        te2_label=None, adapter_label=None,
        show_shift=False, show_flux_shift=False, show_max_seq=False,
        show_hidream_4bit=False, show_min_max_t=False, show_llm_lr=False, show_sdxl_lr=False,
        shift_default=1, min_t_default=0.0, max_t_default=1.0, max_seq_default=256,
        notes="Tentative support. Very high VRAM requirements. Not actively maintained. LoRA in ComfyUI format.",
    ),
    "cosmos_predict2": dict(
        main_label="transformer_path .pt (Cosmos-Predict2-*/model.pt)",
        vae_label="VAE .safetensors (wan_2.1_vae — Wan VAE!)",
        te_label="T5 .safetensors (oldt5_xxl_fp16 — older T5, not standard!)",
        te2_label=None, adapter_label=None,
        show_shift=False, show_flux_shift=False, show_max_seq=False,
        show_hidream_4bit=False, show_min_max_t=False, show_llm_lr=False, show_sdxl_lr=False,
        shift_default=1, min_t_default=0.0, max_t_default=1.0, max_seq_default=256,
        notes="If using fp8, use float8_e5m2 (NOT e4m3fn). T5 must be the older oldt5_xxl version. LoRA in ComfyUI format.",
    ),
    "omnigen2": dict(
        main_label="Diffusers folder (OmniGen2)",
        vae_label=None, te_label=None, te2_label=None, adapter_label=None,
        show_shift=False, show_flux_shift=True, show_max_seq=False,
        show_hidream_4bit=False, show_min_max_t=False, show_llm_lr=False, show_sdxl_lr=False,
        shift_default=1, min_t_default=0.0, max_t_default=1.0, max_seq_default=256,
        notes="Only t2i (single image + caption) training supported. LoRA in ComfyUI format.",
    ),
    "ideogram4": dict(
        main_label="diffusion_model .safetensors (ideogram4_fp8_scaled)",
        vae_label="VAE .safetensors (flux2-vae)",
        te_label="text_encoder .safetensors (qwen3vl_8b_fp8_scaled)",
        te2_label=None, adapter_label=None,
        show_shift=True, show_flux_shift=False, show_max_seq=False,
        show_hidream_4bit=False, show_min_max_t=False, show_llm_lr=False, show_sdxl_lr=False,
        shift_default=3, min_t_default=0.0, max_t_default=1.0, max_seq_default=256,
        notes="shift=3 default. 24GB VRAM sufficient for LoRA. LoRA saved in ComfyUI format.",
    ),
    "ernie_image": dict(
        main_label="diffusion_model .safetensors (ernie-image)",
        vae_label="VAE .safetensors (flux2-vae)",
        te_label="text_encoder .safetensors (ministral-3-3b)",
        te2_label=None, adapter_label=None,
        show_shift=True, show_flux_shift=False, show_max_seq=False,
        show_hidream_4bit=False, show_min_max_t=False, show_llm_lr=False, show_sdxl_lr=False,
        shift_default=3, min_t_default=0.0, max_t_default=1.0, max_seq_default=256,
        notes="shift=3 default. Use ComfyUI-compatible files. LoRA saved in ComfyUI format.",
    ),
    "anima": dict(
        main_label="transformer_path .safetensors (anima-preview)",
        vae_label="VAE .safetensors (qwen_image_vae)",
        te_label="LLM .safetensors (qwen_3_06b_base)",
        te2_label=None, adapter_label=None,
        show_shift=False, show_flux_shift=False, show_max_seq=False,
        show_hidream_4bit=False, show_min_max_t=False, show_llm_lr=True, show_sdxl_lr=False,
        shift_default=1, min_t_default=0.0, max_t_default=1.0, max_seq_default=256,
        notes="Official ComfyUI format files. llm_adapter_lr=0 is safer for small datasets. LoRA in ComfyUI format.",
    ),
    "krea2": dict(
        main_label="diffusion_model .safetensors (krea2_raw)",
        vae_label="VAE .safetensors (qwen_image_vae)",
        te_label="text_encoder .safetensors (qwen3vl_4b_bf16)",
        te2_label=None, adapter_label=None,
        show_shift=False, show_flux_shift=False, show_max_seq=False,
        show_hidream_4bit=False, show_min_max_t=False, show_llm_lr=False, show_sdxl_lr=False,
        shift_default=1, min_t_default=0.0, max_t_default=1.0, max_seq_default=256,
        notes="Rank 32 LoRA at 512 res fits in 24GB VRAM. Use ComfyUI files. LoRA in ComfyUI format.",
    ),
    "auraflow": dict(
        main_label="transformer .safetensors (pony-v7-base or auraflow base)",
        vae_label="VAE .safetensors (sdxl_vae)",
        te_label="text_encoder .safetensors (umt5_auraflow.fp16)",
        te2_label=None, adapter_label=None,
        show_shift=False, show_flux_shift=False, show_max_seq=True,
        show_hidream_4bit=False, show_min_max_t=False, show_llm_lr=False, show_sdxl_lr=False,
        shift_default=1, min_t_default=0.0, max_t_default=1.0, max_seq_default=768,
        notes="max_sequence_length=768 for Pony-V7, 256 for base AuraFlow. LoRA in Diffusers format (works in ComfyUI).",
    ),
    "sd3": dict(
        main_label="Diffusers folder (stable-diffusion-3.5-medium or large)",
        vae_label=None, te_label=None, te2_label=None, adapter_label=None,
        show_shift=False, show_flux_shift=True, show_max_seq=False,
        show_hidream_4bit=False, show_min_max_t=False, show_llm_lr=False, show_sdxl_lr=False,
        shift_default=1, min_t_default=0.0, max_t_default=1.0, max_seq_default=256,
        notes="Tested on SD3.5 Medium and Large. LoRA saved in Diffusers format (works in ComfyUI).",
    ),
    "sdxl": dict(
        main_label="checkpoint .safetensors (sd_xl_base_1.0)",
        vae_label=None, te_label=None, te2_label=None, adapter_label=None,
        show_shift=False, show_flux_shift=False, show_max_seq=False,
        show_hidream_4bit=False, show_min_max_t=False, show_llm_lr=False, show_sdxl_lr=True,
        shift_default=1, min_t_default=0.0, max_t_default=1.0, max_seq_default=256,
        notes="Text encoders are trained (not cached). LoRA in Kohya sd-scripts format. FFT needs 48GB VRAM.",
    ),
}

def _p(rank, res, lr, blocks=0, float8=True, batch=1, grad=1, cache=1, max_seq=None, note=None):
    return dict(rank=rank, res=res, lr=lr, blocks=blocks, float8=float8,
                batch=batch, grad=grad, cache=cache, max_seq=max_seq, note=note)

_IMPOSSIBLE = None  # sentinel — model truly can't run at this VRAM tier

# fmt: off
# VRAM preset tables. Each entry maps model_key → settings dict (or _IMPOSSIBLE).
# For 8/16 GB: impossible models get _IMPOSSIBLE (status note shown, no popup).
# For 16 GB user tried to maximize — only truly impossible models get _IMPOSSIBLE.
VRAM_PRESETS: dict[int, dict[str, dict | None]] = {
    8: {
        "flux":             _p(8,  512, "5e-5", 24, True,  1, 2, 1, note="⚠️ 8GB에서 매우 불안정. OOM 가능성 높음. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 권장"),
        "flux_kontext":     _p(8,  512, "5e-5", 24, True,  1, 2, 1, note="⚠️ 8GB에서 매우 불안정. 이미지 단독 학습만 권장"),
        "flux2_dev":        _IMPOSSIBLE,
        "flux2_klein4b":    _p(8,  512, "1e-4", 14, True,  1, 2, 1, note="⚠️ 8GB에서 불안정. 실패 시 blocks_to_swap 증가"),
        "flux2_klein9b":    _IMPOSSIBLE,
        "chroma":           _p(16, 512, "1e-4", 14, True,  1, 2, 1),
        "qwen_image":       _p(16, 640, "5e-5", 20, True,  1, 2, 1, note="⚠️ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 필수. 640px은 공식 권장 해상도"),
        "qwen_image_edit":  _IMPOSSIBLE,
        "z_image":          _p(32, 512, "1e-4",  0, True,  1, 1, 1),
        "hunyuan_image":    _IMPOSSIBLE,
        "hunyuan_video":    _IMPOSSIBLE,
        "hunyuan_video_15": _IMPOSSIBLE,
        "hidream":          _IMPOSSIBLE,
        "ltx_video":        _p(16, 512, "1e-4", 10, True,  1, 2, 1),
        "ltx2":             _IMPOSSIBLE,
        "wan21":            _p(16, 512, "5e-5", 12, True,  1, 2, 1, note="Wan2.1 1.3B 모델만 가능. 14B는 8GB 불가"),
        "wan22_low":        _IMPOSSIBLE,
        "wan22_high":       _IMPOSSIBLE,
        "lumina_2":         _p(16, 512, "5e-5", 10, False, 1, 2, 1),
        "cosmos":           _IMPOSSIBLE,
        "cosmos_predict2":  _IMPOSSIBLE,
        "omnigen2":         _IMPOSSIBLE,
        "ideogram4":        _IMPOSSIBLE,
        "ernie_image":      _IMPOSSIBLE,
        "anima":            _p(16, 512, "5e-5",  8, False, 1, 1, 1),
        "krea2":            _p(16, 512, "1e-4",  4, True,  1, 1, 1),
        "auraflow":         _p(16, 512, "1e-4",  8, True,  1, 2, 1, max_seq=256),
        "sd3":              _p(16, 512, "1e-4",  8, True,  1, 2, 1),
        "sdxl":             _p(16, 512, "4e-5",  0, False, 1, 2, 1),
    },
    16: {
        "flux":             _p(16, 512, "5e-5", 14, True,  1, 1, 1),
        "flux_kontext":     _p(16, 512, "5e-5", 14, True,  1, 1, 1),
        "flux2_dev":        _p(8,  512, "1e-4", 28, True,  1, 2, 1, note="⚠️ Flux 2 Dev 12B는 16GB에서 매우 느림. blocks_to_swap 최대로 설정. 장시간 소요"),
        "flux2_klein4b":    _p(32, 512, "1e-4",  6, True,  1, 1, 1),
        "flux2_klein9b":    _p(16, 512, "1e-4", 20, True,  1, 2, 1, note="⚠️ 9B 모델. 16GB에서 빡빡함. 실패 시 rank 낮추기"),
        "chroma":           _p(32, 768, "1e-4",  6, True,  1, 1, 1),
        "qwen_image":       _p(32, 768, "5e-5", 10, True,  1, 1, 1, note="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 권장"),
        "qwen_image_edit":  _p(8,  512, "5e-5", 28, True,  1, 2, 1, note="⚠️ 16GB에서 매우 빡빡. expandable_segments:True 필수"),
        "z_image":          _p(64, 768, "1e-4",  0, True,  1, 1, 1),
        "hunyuan_image":    _p(16, 512, "5e-5", 24, True,  1, 2, 1, note="📌 HunyuanImage는 512 입력 → 실질적 1024px 효과. 16GB에서 가능"),
        "hunyuan_video":    _p(4,  256, "5e-5", 38, True,  1, 2, 1, note="⚠️ 16GB에서 이미지 단일 프레임 학습만 권장. 비디오 시퀀스는 OOM 가능. 매우 느림"),
        "hunyuan_video_15": _IMPOSSIBLE,
        "hidream":          _p(8, 1024, "5e-5", 28, True,  1, 2, 1, max_seq=128, note="⚠️ 4bit Llama3 필수. 매우 느림. 실패 시 rank 4 / blocks 32로"),
        "ltx_video":        _p(32, 768, "1e-4",  4, True,  1, 1, 1),
        "ltx2":             _p(8,  512, "1e-4", 44, True,  1, 2, 1, note="⚠️ LTX 2.3 22B는 16GB에서 한계. blocks_to_swap 최대(44). 매우 느림"),
        "wan21":            _p(32, 768, "5e-5",  0, True,  1, 1, 1, note="Wan2.1 1.3B 기준. 14B 사용 시 blocks_to_swap=28 이상 필요"),
        "wan22_low":        _p(16, 512, "5e-5", 28, True,  1, 2, 1, note="⚠️ 14B 모델. 매우 느림. 실패 시 rank 8 / blocks 32"),
        "wan22_high":       _p(16, 512, "5e-5", 28, True,  1, 2, 1, note="⚠️ 14B 모델. 매우 느림. 실패 시 rank 8 / blocks 32"),
        "lumina_2":         _p(32, 768, "5e-5",  4, False, 1, 1, 1),
        "cosmos":           _IMPOSSIBLE,
        "cosmos_predict2":  _p(8,  512, "5e-5", 24, True,  1, 2, 1, note="📌 float8_e5m2 사용 필수 (e4m3fn 아님). 2B 변종 권장"),
        "omnigen2":         _p(16, 512, "5e-5", 14, True,  1, 2, 1),
        "ideogram4":        _p(8,  512, "1e-4", 20, True,  1, 2, 1, note="⚠️ 16GB에서 빡빡함. 실패 시 blocks 증가"),
        "ernie_image":      _p(16, 512, "1e-4", 14, True,  1, 1, 1),
        "anima":            _p(32, 768, "5e-5",  0, False, 1, 1, 1),
        "krea2":            _p(32, 512, "1e-4",  0, True,  1, 1, 1),
        "auraflow":         _p(32, 768, "1e-4",  4, True,  1, 1, 1, max_seq=768),
        "sd3":              _p(32, 768, "1e-4",  0, True,  1, 1, 1),
        "sdxl":             _p(32, 768, "4e-5",  0, False, 1, 1, 1),
    },
    24: {
        "flux":             _p(32, 768, "5e-5",  6, True,  1, 1, 1),
        "flux_kontext":     _p(32, 768, "5e-5",  6, True,  1, 1, 1),
        "flux2_dev":        _p(16, 512, "1e-4", 24, True,  1, 1, 1, note="⚠️ Flux 2 Dev 12B는 24GB에서도 빡빡. 실패 시 blocks 증가"),
        "flux2_klein4b":    _p(64, 768, "1e-4",  0, False, 1, 1, 1),
        "flux2_klein9b":    _p(32, 768, "1e-4", 10, True,  1, 1, 1),
        "chroma":           _p(64,1024, "1e-4",  0, True,  1, 1, 1),
        "qwen_image":       _p(64,1024, "5e-5",  0, True,  1, 1, 1),
        "qwen_image_edit":  _p(16, 512, "5e-5", 16, True,  1, 1, 1),
        "z_image":          _p(64,1024, "1e-4",  0, True,  1, 1, 1),
        "hunyuan_image":    _p(32,1024, "5e-5", 12, True,  1, 1, 1),
        "hunyuan_video":    _p(16, 320, "5e-5", 30, True,  1, 2, 1, note="이미지 단일 프레임 권장. 비디오 시퀀스는 VRAM 추가 필요"),
        "hunyuan_video_15": _p(8,  256, "5e-5", 36, True,  1, 2, 1, note="⚠️ 24GB에서도 매우 어려움. 이미지 전용 권장"),
        "hidream":          _p(16,1024, "5e-5", 20, True,  1, 1, 1, max_seq=128, note="4bit Llama3 권장. 학습 가능"),
        "ltx_video":        _p(64,1024, "1e-4",  0, False, 1, 1, 1),
        "ltx2":             _p(16, 512, "1e-4", 40, True,  1, 1, 1),
        "wan21":            _p(64,1024, "5e-5",  0, True,  1, 1, 1),
        "wan22_low":        _p(32, 768, "5e-5", 20, True,  1, 1, 1),
        "wan22_high":       _p(32, 768, "5e-5", 20, True,  1, 1, 1),
        "lumina_2":         _p(64,1024, "5e-5",  0, False, 1, 1, 1),
        "cosmos":           _p(8,  512, "5e-5",  0, False, 1, 2, 1, note="⚠️ Cosmos는 학습 품질이 불확실. 공식 지원 잠정적"),
        "cosmos_predict2":  _p(32, 768, "5e-5", 10, True,  1, 1, 1, note="float8_e5m2 사용 필수"),
        "omnigen2":         _p(32, 768, "5e-5",  6, True,  1, 1, 1),
        "ideogram4":        _p(32, 512, "1e-4",  6, True,  1, 1, 1),
        "ernie_image":      _p(32, 768, "1e-4",  4, True,  1, 1, 1),
        "anima":            _p(64,1024, "5e-5",  0, False, 1, 1, 1),
        "krea2":            _p(64, 768, "1e-4",  0, True,  1, 1, 1),
        "auraflow":         _p(64,1024, "1e-4",  0, True,  1, 1, 1, max_seq=768),
        "sd3":              _p(64,1024, "1e-4",  0, True,  1, 1, 1),
        "sdxl":             _p(64,1024, "4e-5",  0, False, 1, 1, 1),
    },
    32: {
        "flux":             _p(64,1024, "5e-5",  0, True,  1, 1, 1),
        "flux_kontext":     _p(64,1024, "5e-5",  0, True,  1, 1, 1),
        "flux2_dev":        _p(32, 768, "1e-4", 12, True,  1, 1, 1),
        "flux2_klein4b":    _p(64,1024, "1e-4",  0, False, 1, 1, 1),
        "flux2_klein9b":    _p(64,1024, "1e-4",  0, True,  1, 1, 1),
        "chroma":           _p(64,1024, "1e-4",  0, True,  1, 1, 1),
        "qwen_image":       _p(64,1024, "5e-5",  0, True,  1, 1, 1),
        "qwen_image_edit":  _p(32, 768, "5e-5",  8, True,  1, 1, 1),
        "z_image":          _p(128,1024,"1e-4",  0, True,  1, 1, 1),
        "hunyuan_image":    _p(64,1024, "5e-5",  0, True,  1, 1, 1),
        "hunyuan_video":    _p(16, 512, "5e-5", 20, True,  1, 1, 1),
        "hunyuan_video_15": _p(16, 512, "5e-5", 24, True,  1, 1, 1),
        "hidream":          _p(32,1024, "5e-5",  0, True,  1, 1, 1, max_seq=128),
        "ltx_video":        _p(64,1024, "1e-4",  0, False, 1, 1, 1),
        "ltx2":             _p(32, 768, "1e-4", 30, True,  1, 1, 1),
        "wan21":            _p(64,1024, "5e-5",  0, True,  1, 1, 1),
        "wan22_low":        _p(64,1024, "5e-5", 10, True,  1, 1, 1),
        "wan22_high":       _p(64,1024, "5e-5", 10, True,  1, 1, 1),
        "lumina_2":         _p(64,1024, "5e-5",  0, False, 1, 1, 1),
        "cosmos":           _p(16, 512, "5e-5",  0, False, 1, 1, 1, note="⚠️ Cosmos는 학습 품질 불확실. 잠정적 지원"),
        "cosmos_predict2":  _p(64,1024, "5e-5",  0, True,  1, 1, 1, note="float8_e5m2 사용 필수"),
        "omnigen2":         _p(64,1024, "5e-5",  0, True,  1, 1, 1),
        "ideogram4":        _p(64, 768, "1e-4",  0, True,  1, 1, 1),
        "ernie_image":      _p(64,1024, "1e-4",  0, True,  1, 1, 1),
        "anima":            _p(64,1024, "5e-5",  0, False, 1, 1, 1),
        "krea2":            _p(128,1024,"1e-4",  0, True,  1, 1, 1),
        "auraflow":         _p(64,1024, "1e-4",  0, True,  1, 1, 1, max_seq=768),
        "sd3":              _p(64,1024, "1e-4",  0, True,  1, 1, 1),
        "sdxl":             _p(64,1024, "4e-5",  0, False, 1, 1, 1),
    },
}
# fmt: on


def apply_vram_preset(model_label: str, vram_gb: int):
    gr = require_gradio()
    model_key = _get_model_key(model_label)
    tier = VRAM_PRESETS.get(vram_gb, {})
    p = tier.get(model_key, _IMPOSSIBLE)

    if p is _IMPOSSIBLE:
        msg = (
            f"🚫 {model_label}은(는) {vram_gb}GB VRAM에서 학습이 불가능하거나 "
            "지원되지 않습니다. 더 큰 VRAM 티어를 선택하세요."
        )
        return (
            msg,
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(),
        )

    note = p.get("note") or ""
    msg = f"✅ {model_label} — {vram_gb}GB 프리셋 적용됨."
    if note:
        msg += f"\n{note}"

    max_seq_up = gr.update(value=p["max_seq"]) if p.get("max_seq") is not None else gr.update()

    return (
        msg,
        gr.update(value=p["res"]),
        gr.update(value=p["rank"]),
        gr.update(value=p["lr"]),
        gr.update(value=p["blocks"]),
        gr.update(value=p["float8"]),
        gr.update(value=p["batch"]),
        gr.update(value=p["grad"]),
        max_seq_up,
    )


TRAIN_PROC: subprocess.Popen | None = None
TRAIN_PID: int | None = None
TRAIN_LOG: Path | None = None
TB_PROC: subprocess.Popen | None = None
TB_LOG: Path | None = None


def require_gradio():
    try:
        import gradio as gr
        return gr
    except ImportError as exc:
        raise SystemExit(
            "Gradio is not installed in the diffusion-pipe environment.\n"
            "Install it with:\n"
            "  C:\\ai\\diffusion-pipe\\run_wsl.bat python -m pip install gradio\n"
            "Then run C:\\ai\\diffusion-pipe\\dp_ui.bat again."
        ) from exc


def win_to_wsl(value: str) -> str:
    value = (value or "").strip()
    while len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1].strip()
    value = value.strip('"').strip("'").strip()
    if len(value) >= 3 and value[1:3] in (":/", ":\\"):
        rest = value[3:].replace("\\", "/")
        return f"/mnt/{value[0].lower()}/{rest}"
    return value.replace("\\", "/")


def list_configs() -> list[str]:
    if not CONFIG_ROOT.exists():
        return []
    return [
        p.relative_to(CONFIG_ROOT).as_posix()
        for p in sorted(CONFIG_ROOT.rglob("*.toml"))
        if not p.name.endswith(".dataset.toml")
    ]


def config_path(choice: str) -> Path:
    raw = (choice or "").strip()
    if not raw:
        raise ValueError("Select a config first.")
    path = Path(win_to_wsl(raw))
    if not path.is_absolute():
        path = CONFIG_ROOT / raw
    if not path.is_file():
        raise ValueError(f"Missing config: {path}")
    return path


def quote_toml(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def tail(path: Path | None, lines: int = 80) -> str:
    if path is None or not path.exists():
        return "No log yet."
    data = path.read_text(errors="replace").splitlines()
    return "\n".join(data[-lines:])


def pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def read_train_pid() -> int | None:
    if not TRAIN_PID_FILE.exists():
        return None
    text = TRAIN_PID_FILE.read_text(errors="replace").strip()
    return int(text) if text.isdigit() else None


def write_train_pid(pid: int) -> None:
    LOG_ROOT.mkdir(exist_ok=True)
    TRAIN_PID_FILE.write_text(str(pid), encoding="utf-8")


def process_alive(proc: subprocess.Popen | None) -> bool:
    if pid_alive(TRAIN_PID):
        return True
    if pid_alive(read_train_pid()):
        return True
    return proc is not None and proc.poll() is None


def stop_process(proc: subprocess.Popen | None, pid: int | None = None, force: bool = False) -> str:
    pid = pid or read_train_pid()
    if pid_alive(pid):
        try:
            sig = signal.SIGKILL if force else signal.SIGTERM
            os.killpg(pid, sig)
            if TRAIN_PID_FILE.exists():
                TRAIN_PID_FILE.unlink()
            action = "Force stop" if force else "Stop"
            return f"{action} signal sent to process group {pid}."
        except ProcessLookupError:
            return "Process already stopped."
    if proc is None:
        return "No process was started from this UI."
    if proc.poll() is not None:
        return f"Process already exited with code {proc.returncode}."
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        return "Stop signal sent."
    except ProcessLookupError:
        return "Process already stopped."


def repo_windows_path() -> str:
    try:
        return subprocess.check_output(["wslpath", "-w", str(REPO_DIR)], text=True).strip()
    except Exception:
        return r"C:\ai\diffusion-pipe"


def launch_windows_bat(filename: str, args: list[str] | None = None) -> str:
    bat_path = Path(repo_windows_path()) / filename
    args = args or []
    ps_command = f"Start-Process -FilePath {shlex.quote(str(bat_path))}"
    if args:
        ps_args = ", ".join("'" + arg.replace("'", "''") + "'" for arg in args)
        ps_command += f" -ArgumentList @({ps_args})"
    try:
        subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-Command", ps_command],
            cwd=REPO_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        return f"Failed to launch {filename}: {exc}"
    return (
        f"Launched {filename}. A console window should open immediately. "
        "If it looks quiet, WSL/micromamba is loading and can take 10-30 seconds. "
        "Complete the prompts there, then click Refresh configs."
    )


def launch_dp_wizard() -> str:
    return launch_windows_bat("dp_wizard.bat")


def launch_aitk_converter(config_yaml: str) -> str:
    config_yaml = (config_yaml or "").strip()
    if config_yaml:
        path = Path(win_to_wsl(config_yaml))
        if not path.is_file():
            return (
                "AI-Toolkit config.yaml was not found. "
                "Paste the full path to config.yaml, or leave this field blank to choose it in the console. "
                "AI-Toolkit itself does not need to be installed."
            )
        return launch_windows_bat("AITK_to_diffusion_pipe.bat", ["--config", config_yaml])
    return launch_windows_bat("AITK_to_diffusion_pipe.bat")


def _get_model_key(model_label: str) -> str:
    return MODEL_LABELS.get(model_label, "krea2")


def ui_model_block(
    model_label: str,
    main_path: str,
    vae_path: str,
    text_encoder_path: str,
    text_encoder_2_path: str,
    adapter_path: str,
    use_float8: bool,
    flux_shift: bool,
    shift: int,
    max_sequence_length: int,
    llm_adapter_lr: str,
    unet_lr: str,
    text_encoder_1_lr: str,
    text_encoder_2_lr: str,
    hidream_4bit: bool,
    min_t: float,
    max_t: float,
) -> tuple[str, str]:
    model_key = _get_model_key(model_label)
    main = win_to_wsl(main_path)
    vae = win_to_wsl(vae_path)
    te = win_to_wsl(text_encoder_path)
    te2 = win_to_wsl(text_encoder_2_path)
    adapter = win_to_wsl(adapter_path)
    float8_dm = "diffusion_model_dtype = 'float8'\n" if use_float8 else ""
    float8_tr = "transformer_dtype = 'float8'\n" if use_float8 else ""

    if model_key == "flux":
        tr_line = f"transformer_path = {quote_toml(adapter)}\n" if adapter else ""
        fs_line = "flux_shift = true\n" if flux_shift else ""
        return "flux", f"""[model]
type = 'flux'
diffusers_path = {quote_toml(main)}
{tr_line}dtype = 'bfloat16'
{float8_tr}{fs_line}"""

    if model_key == "flux_kontext":
        tr_line = f"transformer_path = {quote_toml(adapter)}\n" if adapter else ""
        return "flux_kontext", f"""[model]
type = 'flux'
diffusers_path = {quote_toml(main)}
{tr_line}dtype = 'bfloat16'
{float8_tr}"""

    if model_key in ("flux2_dev", "flux2_klein4b", "flux2_klein9b"):
        return "flux2", f"""[model]
type = 'flux2'
diffusion_model = {quote_toml(main)}
vae = {quote_toml(vae)}
text_encoders = [
    {{path = {quote_toml(te)}, type = 'flux2'}}
]
dtype = 'bfloat16'
{float8_dm}timestep_sample_method = 'logit_normal'
shift = {int(shift)}
"""

    if model_key == "chroma":
        tr_line = f"transformer_path = {quote_toml(adapter)}\n" if adapter else ""
        fs_line = "flux_shift = true\n" if flux_shift else ""
        return "chroma", f"""[model]
type = 'chroma'
diffusers_path = {quote_toml(main)}
{tr_line}dtype = 'bfloat16'
{float8_tr}{fs_line}"""

    if model_key == "qwen_image":
        if not te and not vae:
            return "qwen_image", f"""[model]
type = 'qwen_image'
diffusers_path = {quote_toml(main)}
dtype = 'bfloat16'
{float8_tr}timestep_sample_method = 'logit_normal'
"""
        vae_line = f"vae_path = {quote_toml(vae)}\n" if vae else ""
        te_line = f"text_encoder_path = {quote_toml(te)}\n" if te else ""
        return "qwen_image", f"""[model]
type = 'qwen_image'
transformer_path = {quote_toml(main)}
{te_line}{vae_line}dtype = 'bfloat16'
{float8_tr}timestep_sample_method = 'logit_normal'
"""

    if model_key == "qwen_image_edit":
        tr_line = f"transformer_path = {quote_toml(adapter)}\n" if adapter else ""
        return "qwen_image_edit", f"""[model]
type = 'qwen_image'
diffusers_path = {quote_toml(main)}
{tr_line}dtype = 'bfloat16'
{float8_tr}timestep_sample_method = 'logit_normal'
"""

    if model_key == "z_image":
        merge = f"merge_adapters = [{quote_toml(adapter)}]\n" if adapter else ""
        return "z_image", f"""[model]
type = 'z_image'
diffusion_model = {quote_toml(main)}
vae = {quote_toml(vae)}
text_encoders = [
    {{path = {quote_toml(te)}, type = 'lumina2'}}
]
{merge}dtype = 'bfloat16'
{float8_dm}"""

    if model_key == "hunyuan_image":
        return "hunyuan_image", f"""[model]
type = 'hunyuan_image'
transformer_path = {quote_toml(main)}
vae_path = {quote_toml(vae)}
text_encoder_path = {quote_toml(te)}
byt5_path = {quote_toml(te2)}
dtype = 'bfloat16'
{float8_tr}"""

    if model_key == "hunyuan_video":
        return "hunyuan_video", f"""[model]
type = 'hunyuan-video'
transformer_path = {quote_toml(main)}
vae_path = {quote_toml(vae)}
llm_path = {quote_toml(te)}
clip_path = {quote_toml(te2)}
dtype = 'bfloat16'
{float8_tr}timestep_sample_method = 'logit_normal'
"""

    if model_key == "hunyuan_video_15":
        return "hunyuan_video_15", f"""[model]
type = 'hunyuan_video_15'
diffusion_model = {quote_toml(main)}
vae = {quote_toml(vae)}
text_encoders = [
    {{paths = [
        {quote_toml(te)},
        {quote_toml(te2)},
    ], type = 'hunyuan_video_15'}},
]
dtype = 'bfloat16'
{float8_dm}timestep_sample_method = 'logit_normal'
"""

    if model_key == "hidream":
        bit4_line = "llama3_4bit = true\n" if hidream_4bit else ""
        fs_line = "flux_shift = true\n" if flux_shift else ""
        return "hidream", f"""[model]
type = 'hidream'
diffusers_path = {quote_toml(main)}
llama3_path = {quote_toml(te)}
{bit4_line}dtype = 'bfloat16'
{float8_tr}max_llama3_sequence_length = {int(max_sequence_length)}
{fs_line}"""

    if model_key == "ltx_video":
        sf_line = f"single_file_path = {quote_toml(adapter)}\n" if adapter else ""
        return "ltx_video", f"""[model]
type = 'ltx-video'
diffusers_path = {quote_toml(main)}
{sf_line}dtype = 'bfloat16'
{float8_tr}timestep_sample_method = 'logit_normal'
"""

    if model_key == "ltx2":
        return "ltx2", f"""[model]
type = 'ltx2'
diffusion_model = {quote_toml(main)}
text_encoder = {quote_toml(te)}
dtype = 'bfloat16'
{float8_dm}timestep_sample_method = 'logit_normal'
shift = {int(shift)}
"""

    if model_key in ("wan21", "wan22_low", "wan22_high"):
        tr_line = f"transformer_path = {quote_toml(vae)}\n" if vae else ""
        llm_line = f"llm_path = {quote_toml(te)}\n" if te else ""
        float8_wan = "transformer_dtype = 'float8'\n" if use_float8 else ""
        t_range = ""
        if model_key in ("wan22_low", "wan22_high"):
            t_range = f"min_t = {float(min_t)}\nmax_t = {float(max_t)}\n"
        folder = "wan21" if model_key == "wan21" else "wan22"
        return folder, f"""[model]
type = 'wan'
ckpt_path = {quote_toml(main)}
{tr_line}{llm_line}dtype = 'bfloat16'
{float8_wan}timestep_sample_method = 'logit_normal'
{t_range}"""

    if model_key == "lumina_2":
        return "lumina_2", f"""[model]
type = 'lumina_2'
transformer_path = {quote_toml(main)}
llm_path = {quote_toml(te)}
vae_path = {quote_toml(vae)}
dtype = 'bfloat16'
lumina_shift = true
"""

    if model_key == "cosmos":
        return "cosmos", f"""[model]
type = 'cosmos'
transformer_path = {quote_toml(main)}
vae_path = {quote_toml(vae)}
text_encoder_path = {quote_toml(te)}
dtype = 'bfloat16'
"""

    if model_key == "cosmos_predict2":
        return "cosmos_predict2", f"""[model]
type = 'cosmos_predict2'
transformer_path = {quote_toml(main)}
vae_path = {quote_toml(vae)}
t5_path = {quote_toml(te)}
dtype = 'bfloat16'
{float8_tr}"""

    if model_key == "omnigen2":
        fs_line = "flux_shift = true\n" if flux_shift else ""
        return "omnigen2", f"""[model]
type = 'omnigen2'
diffusers_path = {quote_toml(main)}
dtype = 'bfloat16'
{fs_line}"""

    if model_key == "ideogram4":
        return "ideogram4", f"""[model]
type = 'ideogram4'
diffusion_model = {quote_toml(main)}
vae = {quote_toml(vae)}
text_encoders = [
    {{path = {quote_toml(te)}, type = 'ideogram4'}}
]
dtype = 'bfloat16'
{float8_dm}timestep_sample_method = 'logit_normal'
shift = {int(shift)}
"""

    if model_key == "ernie_image":
        return "ernie_image", f"""[model]
type = 'ernie_image'
diffusion_model = {quote_toml(main)}
vae = {quote_toml(vae)}
text_encoders = [
    {{path = {quote_toml(te)}, type = 'flux2'}}
]
dtype = 'bfloat16'
{float8_dm}timestep_sample_method = 'logit_normal'
shift = {int(shift)}
"""

    if model_key == "anima":
        return "anima", f"""[model]
type = 'anima'
transformer_path = {quote_toml(main)}
vae_path = {quote_toml(vae)}
llm_path = {quote_toml(te)}
dtype = 'bfloat16'
llm_adapter_lr = {llm_adapter_lr or "0"}
"""

    if model_key == "krea2":
        return "krea2", f"""[model]
type = 'krea2'
diffusion_model = {quote_toml(main)}
vae = {quote_toml(vae)}
text_encoders = [
    {{path = {quote_toml(te)}, type = 'krea2'}}
]
dtype = 'bfloat16'
{float8_dm}timestep_sample_method = 'logit_normal'
"""

    if model_key == "auraflow":
        return "auraflow", f"""[model]
type = 'auraflow'
transformer_path = {quote_toml(main)}
text_encoder_path = {quote_toml(te)}
vae_path = {quote_toml(vae)}
dtype = 'bfloat16'
{float8_tr}timestep_sample_method = 'logit_normal'
max_sequence_length = {int(max_sequence_length)}
"""

    if model_key == "sd3":
        fs_line = "flux_shift = true\n" if flux_shift else ""
        return "sd3", f"""[model]
type = 'sd3'
diffusers_path = {quote_toml(main)}
dtype = 'bfloat16'
{float8_tr}{fs_line}"""

    if model_key == "sdxl":
        return "sdxl", f"""[model]
type = 'sdxl'
checkpoint_path = {quote_toml(main)}
dtype = 'bfloat16'
unet_lr = {unet_lr or "4e-5"}
text_encoder_1_lr = {text_encoder_1_lr or "2e-5"}
text_encoder_2_lr = {text_encoder_2_lr or "2e-5"}
"""

    raise ValueError(f"Unsupported model: {model_label}")


def generate_config_from_ui(
    model_label: str,
    run_name: str,
    dataset_path: str,
    output_folder: str,
    trigger_word: str,
    main_model_path: str,
    vae_path: str,
    text_encoder_path: str,
    text_encoder_2_path: str,
    adapter_path: str,
    resolution: int,
    min_ar: float,
    max_ar: float,
    ar_buckets: int,
    repeats: int,
    rank: int,
    learning_rate: str,
    max_steps: int,
    save_every: int,
    batch_size: int,
    grad_accum: int,
    blocks_to_swap: int,
    cache_batch: int,
    optimizer: str,
    use_float8: bool,
    flux_shift: bool,
    shift: int,
    max_sequence_length: int,
    llm_adapter_lr: str,
    unet_lr: str,
    text_encoder_1_lr: str,
    text_encoder_2_lr: str,
    hidream_4bit: bool,
    min_t: float,
    max_t: float,
):
    import gradio as gr

    def _cfg_update():
        return gr.update(choices=list_configs())

    if not run_name.strip():
        return "Run name is required.", _cfg_update()
    if not dataset_path.strip():
        return "Dataset path is required.", _cfg_update()
    if not main_model_path.strip():
        return "Main model path/folder is required.", _cfg_update()

    dataset = Path(win_to_wsl(dataset_path))
    if not dataset.is_dir():
        return f"Dataset folder not found: {dataset}", _cfg_update()

    model_key, model_text = ui_model_block(
        model_label, main_model_path, vae_path, text_encoder_path, text_encoder_2_path,
        adapter_path, use_float8, flux_shift, int(shift), int(max_sequence_length),
        llm_adapter_lr, unet_lr, text_encoder_1_lr, text_encoder_2_lr,
        hidream_4bit, float(min_t), float(max_t),
    )

    output_dir = win_to_wsl(output_folder.strip() or f"{RUN_ROOT.as_posix()}/{model_key}/{run_name}")
    config_dir = CONFIG_ROOT / model_key
    dataset_config = config_dir / f"{run_name}.dataset.toml"
    train_config = config_dir / f"{run_name}.toml"
    caption_prefix = f"{trigger_word.strip()}, " if trigger_word.strip() else ""

    dataset_text = f"""resolutions = [{int(resolution)}]
enable_ar_bucket = true
min_ar = {float(min_ar)}
max_ar = {float(max_ar)}
num_ar_buckets = {int(ar_buckets)}
frame_buckets = [1]
skip_empty_caption = false

[[directory]]
path = {quote_toml(dataset.as_posix())}
num_repeats = {int(repeats)}
caption_prefix = {quote_toml(caption_prefix)}
"""

    optimizer = (optimizer or "adamw8bitkahan").lower()
    if optimizer == "adamw_optimi":
        optimizer_text = f"""[optimizer]
type = 'adamw_optimi'
lr = {learning_rate}
betas = [0.9, 0.99]
weight_decay = 0.0001
eps = 1e-8
"""
    else:
        optimizer_text = f"""[optimizer]
type = 'AdamW8bitKahan'
lr = {learning_rate}
betas = [0.9, 0.99]
weight_decay = 0.0001
stabilize = false
"""

    blocks_line = f"blocks_to_swap = {int(blocks_to_swap)}\n" if int(blocks_to_swap) > 0 else ""
    train_text = f"""output_dir = {quote_toml(output_dir)}
dataset = {quote_toml(dataset_config.as_posix())}

epochs = 1000
max_steps = {int(max_steps)}
micro_batch_size_per_gpu = {int(batch_size)}
pipeline_stages = 1
gradient_accumulation_steps = {int(grad_accum)}
gradient_clipping = 1.0
warmup_steps = 100
eval_before_first_step = false
eval_every_n_steps = {max(int(save_every), 1)}
eval_micro_batch_size_per_gpu = 1
eval_gradient_accumulation_steps = 1
save_every_n_steps = {int(save_every)}
checkpoint_every_n_minutes = 120
activation_checkpointing = true
partition_method = 'parameters'
save_dtype = 'bfloat16'
caching_batch_size = {int(cache_batch)}
steps_per_print = 1
{blocks_line}
{model_text}
[adapter]
type = 'lora'
rank = {int(rank)}
dtype = 'bfloat16'
dropout = 0.0

{optimizer_text}
[monitoring]
enable_wandb = false
wandb_api_key = ''
wandb_tracker_name = ''
wandb_run_name = {quote_toml(run_name)}
"""

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    dataset_config.write_text(dataset_text, encoding="utf-8", newline="\n")
    train_config.write_text(train_text, encoding="utf-8", newline="\n")
    return (
        "Generated configs:\n"
        f"  dataset: {dataset_config}\n"
        f"  train:   {train_config}\n"
        "Select it from Config, or click Refresh configs.",
        gr.update(choices=list_configs()),
    )


def latest_train_log() -> Path | None:
    if not LOG_ROOT.exists():
        return None
    logs = sorted(LOG_ROOT.glob("train_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def attach_latest_log() -> tuple[str, str]:
    global TRAIN_LOG, TRAIN_PID
    TRAIN_LOG = latest_train_log()
    TRAIN_PID = read_train_pid()
    if TRAIN_LOG is None:
        return "No UI training log found.", "No log yet."
    pid_text = f" PID: {TRAIN_PID}" if TRAIN_PID else ""
    alive_text = " alive" if pid_alive(TRAIN_PID) else ""
    return f"Attached log: {TRAIN_LOG}{pid_text}{alive_text}", tail(TRAIN_LOG)


def start_training(choice: str, action: str, checkpoint: str) -> tuple[str, str]:
    global TRAIN_PROC, TRAIN_PID, TRAIN_LOG
    if process_alive(TRAIN_PROC):
        return "A training/cache job is already running from this UI.", tail(TRAIN_LOG)

    try:
        cfg = config_path(choice)
    except ValueError as exc:
        return str(exc), tail(TRAIN_LOG)

    extra: list[str] = []
    if action == "cache_only":
        extra = ["--cache_only"]
    elif action == "resume latest checkpoint":
        extra = ["--resume_from_checkpoint"]
    elif action == "resume specific checkpoint":
        ckpt = (checkpoint or "").strip()
        if not ckpt:
            return "Enter a checkpoint folder name for resume specific checkpoint.", tail(TRAIN_LOG)
        extra = ["--resume_from_checkpoint", ckpt]

    LOG_ROOT.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    TRAIN_LOG = LOG_ROOT / f"train_{stamp}.log"
    cmd = ["deepspeed", "--num_gpus=1", "train.py", "--deepspeed", "--config", cfg.as_posix(), *extra]
    TRAIN_LOG.write_text("Running detached:\n" + " ".join(cmd) + "\n\n", encoding="utf-8")
    quoted_cmd = " ".join(shlex.quote(part) for part in cmd)
    quoted_log = shlex.quote(TRAIN_LOG.as_posix())
    launcher = f"nohup setsid {quoted_cmd} >> {quoted_log} 2>&1 < /dev/null & echo $!"
    TRAIN_PROC = subprocess.Popen(
        ["bash", "-lc", launcher],
        cwd=REPO_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = TRAIN_PROC.communicate(timeout=10)
    pid_text = stdout.strip().splitlines()[-1] if stdout.strip() else ""
    if not pid_text.isdigit():
        return f"Failed to start detached job. {stderr.strip()}", tail(TRAIN_LOG)
    TRAIN_PID = int(pid_text)
    write_train_pid(TRAIN_PID)
    return f"Started detached PID {TRAIN_PID}. Closing the Gradio/browser window will not stop this job. Log: {TRAIN_LOG}", tail(TRAIN_LOG)


def stop_training() -> tuple[str, str]:
    return stop_process(TRAIN_PROC, TRAIN_PID), tail(TRAIN_LOG)


def force_stop_training() -> tuple[str, str]:
    return stop_process(TRAIN_PROC, TRAIN_PID, force=True), tail(TRAIN_LOG)


def tensorboard_frame(port: str) -> str:
    port = port if (port or "").isdigit() else "6006"
    return (
        f'<div style="height:760px;border:1px solid #ddd;border-radius:8px;overflow:hidden;">'
        f'<iframe src="http://localhost:{port}" '
        f'style="width:100%;height:100%;border:0;background:white;"></iframe>'
        f'</div>'
    )


def tb_alive() -> bool:
    return TB_PROC is not None and TB_PROC.poll() is None


def start_tensorboard(logdir: str, port: str) -> tuple[str, str]:
    global TB_PROC, TB_LOG
    if tb_alive():
        port = port if (port or "").isdigit() else "6006"
        return f"TensorBoard is already running: http://localhost:{port}", tensorboard_frame(port)

    path = Path(win_to_wsl(logdir or str(RUN_ROOT)))
    if not path.exists():
        return f"Missing log directory: {path}", ""
    port = port if (port or "").isdigit() else "6006"
    LOG_ROOT.mkdir(exist_ok=True)
    TB_LOG = LOG_ROOT / f"tensorboard_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_file = TB_LOG.open("w", encoding="utf-8", errors="replace")
    cmd = ["tensorboard", "--logdir", path.as_posix(), "--host", "0.0.0.0", "--port", port]
    TB_PROC = subprocess.Popen(
        cmd,
        cwd=REPO_DIR,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )
    return f"Started TensorBoard: http://localhost:{port}", tensorboard_frame(port)


def stop_tensorboard() -> tuple[str, str]:
    global TB_PROC
    if TB_PROC is not None and TB_PROC.poll() is None:
        try:
            os.killpg(TB_PROC.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        TB_PROC = None
        return "TensorBoard stopped.", ""
    TB_PROC = None
    return "TensorBoard was not running.", ""


def gpu_status() -> str:
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu",
        "--format=csv",
    ]
    try:
        gpu = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        gpu = f"nvidia-smi failed: {exc}"

    try:
        ps = subprocess.check_output(
            "ps -eo pid,ppid,stat,pcpu,pmem,etime,args | grep -E 'train.py|deepspeed|tensorboard' | grep -v grep",
            shell=True,
            cwd=REPO_DIR,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError:
        ps = "No diffusion-pipe training/tensorboard process found."
    return gpu + "\n\n" + ps


def recent_runs() -> str:
    if not RUN_ROOT.exists():
        return "No training_runs folder yet."
    items = sorted(RUN_ROOT.rglob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    lines = []
    for path in items[:40]:
        try:
            rel = path.relative_to(RUN_ROOT).as_posix()
            size = path.stat().st_size if path.is_file() else 0
            when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))
            lines.append(f"{when}  {size:>10}  {rel}")
        except OSError:
            pass
    return "\n".join(lines) if lines else "No run files yet."


def saved_checkpoints(choice: str) -> str:
    try:
        cfg = config_path(choice)
    except ValueError as exc:
        return str(exc)

    try:
        with cfg.open("rb") as handle:
            config_data = tomllib.load(handle)
    except Exception as exc:
        return f"Could not read config: {exc}"

    output_dir = Path(config_data.get("output_dir", RUN_ROOT))
    save_every = config_data.get("save_every_n_steps")
    max_steps = config_data.get("max_steps")

    if not output_dir.exists():
        return f"Output directory does not exist yet:\n{output_dir}"

    run_dirs = [p for p in output_dir.iterdir() if p.is_dir()]
    if any(p.name.startswith("step") for p in run_dirs):
        run_dirs = [output_dir]
    else:
        run_dirs = sorted(run_dirs, key=lambda p: p.stat().st_mtime, reverse=True)

    rows: list[tuple[int, Path, Path | None]] = []
    for run_dir in run_dirs[:8]:
        for step_dir in run_dir.iterdir():
            if not step_dir.is_dir() or not step_dir.name.startswith("step"):
                continue
            step_text = step_dir.name.removeprefix("step")
            if not step_text.isdigit():
                continue
            adapter = step_dir / "adapter_model.safetensors"
            rows.append((int(step_text), step_dir, adapter if adapter.exists() else None))

    rows.sort(key=lambda item: item[0], reverse=True)
    if not rows:
        return f"No saved step folders found yet.\nOutput directory:\n{output_dir}"

    latest_step = rows[0][0]
    header = [f"Output: {output_dir}"]
    if save_every:
        next_step = latest_step + int(save_every)
        if max_steps:
            next_step = min(next_step, int(max_steps))
        header.append(f"Latest saved step: {latest_step}")
        header.append(f"Save every: {save_every} steps")
        if not max_steps or latest_step < int(max_steps):
            header.append(f"Next expected save: step{next_step}")
    if max_steps:
        header.append(f"Max steps: {max_steps}")

    lines = ["\n".join(header), "", "Saved checkpoints:"]
    for step, step_dir, adapter in rows[:30]:
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(step_dir.stat().st_mtime))
        if adapter:
            size_mb = adapter.stat().st_size / (1024 * 1024)
            adapter_text = f"{size_mb:.1f} MB"
        else:
            adapter_text = "adapter missing"
        rel = step_dir.relative_to(output_dir).as_posix()
        lines.append(f"step{step:<5}  {when}  {adapter_text:>12}  {rel}")
    return "\n".join(lines)


def _run_ps(script: str, timeout: int = 120) -> str:
    """Run a PowerShell script from WSL and return stdout text."""
    try:
        r = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=timeout,
        )
        return (r.stdout or "").strip()
    except Exception:
        return ""


def browse_folder(current: str = "") -> str:
    """Open a Windows FolderBrowserDialog and return the selected path."""
    ps = r"""
Add-Type -AssemblyName System.Windows.Forms
$d = New-Object System.Windows.Forms.FolderBrowserDialog
$d.Description = "폴더를 선택하세요"
try { $d.UseDescriptionForTitle = $true } catch {}
$d.ShowNewFolderButton = $true
$r = $d.ShowDialog()
if ($r -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $d.SelectedPath }
"""
    result = _run_ps(ps)
    return result if result else current


def browse_file(current: str = "", filter_str: str = "All files (*.*)|*.*") -> str:
    """Open a Windows OpenFileDialog and return the selected file path."""
    ps = f"""
Add-Type -AssemblyName System.Windows.Forms
$d = New-Object System.Windows.Forms.OpenFileDialog
$d.Title = "파일을 선택하세요"
$d.Filter = "{filter_str}"
$d.CheckFileExists = $true
$r = $d.ShowDialog()
if ($r -eq [System.Windows.Forms.DialogResult]::OK) {{ Write-Output $d.FileName }}
"""
    result = _run_ps(ps)
    return result if result else current


_SF = "SafeTensors (*.safetensors)|*.safetensors|All files (*.*)|*.*"
_SF_GGUF = "Model files (*.safetensors;*.gguf)|*.safetensors;*.gguf|All files (*.*)|*.*"


def refresh_configs():
    import gradio as gr
    return gr.update(choices=list_configs())


def _infer_model_label_from_cfg(folder: str, ms: dict) -> str:
    _KEY_TO_LABEL = {v: k for k, v in MODEL_LABELS.items()}
    folder_map = {
        "chroma": "chroma", "flux2": "flux2_dev", "qwen_image": "qwen_image",
        "z_image": "z_image", "hunyuan_image": "hunyuan_image",
        "hunyuan_video": "hunyuan_video", "hunyuan_video_15": "hunyuan_video_15",
        "hidream": "hidream", "ltx_video": "ltx_video", "ltx2": "ltx2",
        "wan21": "wan21", "wan22": "wan22_low",
        "lumina_2": "lumina_2", "cosmos": "cosmos",
        "cosmos_predict2": "cosmos_predict2", "omnigen2": "omnigen2",
        "ideogram4": "ideogram4", "ernie_image": "ernie_image",
        "anima": "anima", "krea2": "krea2", "auraflow": "auraflow",
        "sd3": "sd3", "sdxl": "sdxl",
    }
    model_key = folder_map.get(folder, "krea2")
    if folder == "flux":
        tr = ms.get("transformer_path", "").lower()
        model_key = "flux_kontext" if "kontext" in tr else "flux"
    elif folder == "flux2":
        tes = ms.get("text_encoders") or []
        te_p = (tes[0].get("path", "") if tes else "").lower()
        if "qwen_3_8b" in te_p or "qwen3_8b" in te_p:
            model_key = "flux2_klein9b"
        elif "qwen_3_4b" in te_p or "qwen3_4b" in te_p:
            model_key = "flux2_klein4b"
        else:
            model_key = "flux2_dev"
    elif folder == "qwen_image":
        if ms.get("transformer_path"):
            model_key = "qwen_image_edit"
    elif folder == "wan22":
        model_key = "wan22_high" if float(ms.get("min_t", 0)) >= 0.5 else "wan22_low"
    return _KEY_TO_LABEL.get(model_key, "Krea 2")


def _extract_paths(model_key: str, ms: dict) -> tuple[str, str, str, str, str]:
    """Return (main, vae, te, te2, adapter) from a loaded model TOML section."""
    tes = ms.get("text_encoders") or []

    def _tp(idx=0):
        if not tes or idx >= len(tes):
            return ""
        e = tes[idx]
        if isinstance(e, dict):
            paths = e.get("paths") or [e.get("path", "")]
            return paths[0] if paths else ""
        return ""

    def _tp2(idx=0):
        if not tes or idx >= len(tes):
            return ""
        e = tes[idx]
        if isinstance(e, dict):
            paths = e.get("paths", [])
            return paths[1] if len(paths) > 1 else ""
        return ""

    if model_key in ("flux", "flux_kontext"):
        return ms.get("diffusers_path", ""), "", "", "", ms.get("transformer_path", "")
    if model_key == "chroma":
        return ms.get("diffusers_path", ""), "", "", "", ms.get("transformer_path", "")
    if model_key in ("flux2_dev", "flux2_klein4b", "flux2_klein9b"):
        return ms.get("diffusion_model", ""), ms.get("vae", ""), _tp(), "", ""
    if model_key == "qwen_image":
        main = ms.get("diffusers_path") or ms.get("transformer_path", "")
        return main, ms.get("vae_path", ""), ms.get("text_encoder_path", ""), "", ""
    if model_key == "qwen_image_edit":
        return ms.get("diffusers_path", ""), "", "", "", ms.get("transformer_path", "")
    if model_key == "z_image":
        adps = ms.get("merge_adapters") or []
        return ms.get("diffusion_model", ""), ms.get("vae", ""), _tp(), "", adps[0] if adps else ""
    if model_key == "hunyuan_image":
        return ms.get("transformer_path", ""), ms.get("vae_path", ""), ms.get("text_encoder_path", ""), ms.get("byt5_path", ""), ""
    if model_key == "hunyuan_video":
        return ms.get("transformer_path", ""), ms.get("vae_path", ""), ms.get("llm_path", ""), ms.get("clip_path", ""), ""
    if model_key == "hunyuan_video_15":
        return ms.get("diffusion_model", ""), ms.get("vae", ""), _tp(), _tp2(), ""
    if model_key == "hidream":
        return ms.get("diffusers_path", ""), "", ms.get("llama3_path", ""), "", ""
    if model_key == "ltx_video":
        return ms.get("diffusers_path", ""), "", "", "", ms.get("single_file_path", "")
    if model_key == "ltx2":
        return ms.get("diffusion_model", ""), "", ms.get("text_encoder", ""), "", ""
    if model_key in ("wan21", "wan22_low", "wan22_high"):
        return ms.get("ckpt_path", ""), ms.get("transformer_path", ""), ms.get("llm_path", ""), "", ""
    if model_key == "lumina_2":
        return ms.get("transformer_path", ""), ms.get("vae_path", ""), ms.get("llm_path", ""), "", ""
    if model_key == "cosmos":
        return ms.get("transformer_path", ""), ms.get("vae_path", ""), ms.get("text_encoder_path", ""), "", ""
    if model_key == "cosmos_predict2":
        return ms.get("transformer_path", ""), ms.get("vae_path", ""), ms.get("t5_path", ""), "", ""
    if model_key == "omnigen2":
        return ms.get("diffusers_path", ""), "", "", "", ""
    if model_key in ("ideogram4", "ernie_image"):
        return ms.get("diffusion_model", ""), ms.get("vae", ""), _tp(), "", ""
    if model_key == "anima":
        return ms.get("transformer_path", ""), ms.get("vae_path", ""), ms.get("llm_path", ""), "", ""
    if model_key == "krea2":
        return ms.get("diffusion_model", ""), ms.get("vae", ""), _tp(), "", ""
    if model_key == "auraflow":
        return ms.get("transformer_path", ""), ms.get("vae_path", ""), ms.get("text_encoder_path", ""), "", ""
    if model_key == "sd3":
        return ms.get("diffusers_path", ""), "", "", "", ""
    if model_key == "sdxl":
        return ms.get("checkpoint_path", ""), "", "", "", ""
    return "", "", "", "", ""


def load_config_to_ui(choice: str):
    import gradio as gr
    no = gr.update()
    _N = 41  # outputs after (msg, accordion); 4 extra for vae/te/te2/adapter col visibility

    if not choice:
        return ("",  gr.update()) + (no,) * _N

    try:
        cfg = config_path(choice)
    except ValueError as exc:
        return (str(exc), gr.update()) + (no,) * _N

    try:
        with cfg.open("rb") as f:
            data = tomllib.load(f)
    except Exception as exc:
        return (f"Config 파싱 실패: {exc}", gr.update()) + (no,) * _N

    ms = data.get("model", {})
    opt = data.get("optimizer", {})
    adp = data.get("adapter", {})
    mon = data.get("monitoring", {})

    folder = cfg.parent.name
    label = _infer_model_label_from_cfg(folder, ms)
    model_key = MODEL_LABELS.get(label, "krea2")
    info = MODEL_UI.get(model_key, MODEL_UI["krea2"])

    main_p, vae_p, te_p, te2_p, adp_p = _extract_paths(model_key, ms)

    run_name = mon.get("wandb_run_name") or cfg.stem
    output_d = data.get("output_dir", "")
    rank = adp.get("rank", 32)
    lr_v = str(opt.get("lr", "1e-4"))
    opt_type = opt.get("type", "")
    optimizer = "adamw_optimi" if "optimi" in opt_type.lower() else "adamw8bitkahan"
    max_steps = data.get("max_steps", 1000)
    save_every = data.get("save_every_n_steps", 250)
    batch = data.get("micro_batch_size_per_gpu", 1)
    grad_accum = data.get("gradient_accumulation_steps", 1)
    blocks = data.get("blocks_to_swap", 0)
    cache_batch = data.get("caching_batch_size", 1)
    float8 = ms.get("transformer_dtype") == "float8" or ms.get("diffusion_model_dtype") == "float8"
    flux_shift = bool(ms.get("flux_shift", False))
    shift_v = ms.get("shift", info["shift_default"])
    max_seq = ms.get("max_llama3_sequence_length") or ms.get("max_sequence_length") or info["max_seq_default"]
    hidream_4bit = bool(ms.get("llama3_4bit", False))
    min_t_v = float(ms.get("min_t", info["min_t_default"]))
    max_t_v = float(ms.get("max_t", info["max_t_default"]))
    llm_lr = str(ms.get("llm_adapter_lr", "0"))
    unet_lr = str(ms.get("unet_lr", "4e-5"))
    te1_lr = str(ms.get("text_encoder_1_lr", "2e-5"))
    te2_lr_s = str(ms.get("text_encoder_2_lr", "2e-5"))

    ds_path = Path(data.get("dataset", "")) if data.get("dataset") else None
    dataset_dir = ""
    trigger = ""
    resolution = 512
    min_ar = 0.5
    max_ar = 2.0
    ar_buckets = 7
    repeats = 1

    if ds_path and ds_path.is_file():
        try:
            with ds_path.open("rb") as f:
                ds = tomllib.load(f)
            resolution = (ds.get("resolutions") or [512])[0]
            min_ar = ds.get("min_ar", 0.5)
            max_ar = ds.get("max_ar", 2.0)
            ar_buckets = ds.get("num_ar_buckets", 7)
            dirs = ds.get("directory") or []
            if dirs:
                d0 = dirs[0]
                dataset_dir = d0.get("path", "")
                repeats = d0.get("num_repeats", 1)
                cp = d0.get("caption_prefix", "")
                trigger = cp.rstrip(", ").rstrip(",").strip()
        except Exception:
            pass

    msg = f"✅ 불러옴: {cfg.name}"
    if ds_path and not ds_path.is_file():
        msg += f"\n⚠️ 데이터셋 config 없음: {ds_path}"

    return (
        msg,
        gr.update(open=True),
        gr.update(value=label),
        gr.update(value=run_name),
        gr.update(value=trigger),
        gr.update(value=dataset_dir),
        gr.update(value=output_d),
        gr.update(value=main_p,  label=info["main_label"]),
        gr.update(visible=info["vae_label"]     is not None),   # ui_vae_col
        gr.update(value=vae_p,   label=info["vae_label"]     or "VAE path"),
        gr.update(visible=info["te_label"]      is not None),   # ui_te_col
        gr.update(value=te_p,    label=info["te_label"]      or "Text encoder / LLM path"),
        gr.update(visible=info["te2_label"]     is not None),   # ui_te2_col
        gr.update(value=te2_p,   label=info["te2_label"]     or "Secondary text encoder"),
        gr.update(visible=info["adapter_label"] is not None),   # ui_adapter_col
        gr.update(value=adp_p,   label=info["adapter_label"] or "Adapter / extra path"),
        gr.update(value=f"ℹ️ **{info['notes']}**"),
        gr.update(value=resolution),
        gr.update(value=min_ar),
        gr.update(value=max_ar),
        gr.update(value=ar_buckets),
        gr.update(value=repeats),
        gr.update(value=rank),
        gr.update(value=lr_v),
        gr.update(value=max_steps),
        gr.update(value=save_every),
        gr.update(value=batch),
        gr.update(value=grad_accum),
        gr.update(value=blocks),
        gr.update(value=cache_batch),
        gr.update(value=optimizer),
        gr.update(value=float8),
        gr.update(value=flux_shift,   visible=info["show_flux_shift"]),
        gr.update(value=shift_v,      visible=info["show_shift"]),
        gr.update(value=max_seq,      visible=info["show_max_seq"]),
        gr.update(value=hidream_4bit, visible=info["show_hidream_4bit"]),
        gr.update(value=min_t_v,      visible=info["show_min_max_t"]),
        gr.update(value=max_t_v,      visible=info["show_min_max_t"]),
        gr.update(value=llm_lr,       visible=info["show_llm_lr"]),
        gr.update(visible=info["show_sdxl_lr"]),
        gr.update(value=unet_lr),
        gr.update(value=te1_lr),
        gr.update(value=te2_lr_s),
    )


def build_ui():
    gr = require_gradio()

    def update_model_ui(model_label):
        key = _get_model_key(model_label)
        info = MODEL_UI.get(key, MODEL_UI["krea2"])
        return [
            gr.update(label=info["main_label"]),
            gr.update(visible=info["vae_label"] is not None),
            gr.update(label=info["vae_label"] or "VAE path"),
            gr.update(visible=info["te_label"] is not None),
            gr.update(label=info["te_label"] or "Text encoder / LLM path"),
            gr.update(visible=info["te2_label"] is not None),
            gr.update(label=info["te2_label"] or "Secondary text encoder"),
            gr.update(visible=info["adapter_label"] is not None),
            gr.update(label=info["adapter_label"] or "Adapter / extra path"),
            gr.update(visible=info["show_shift"], value=info["shift_default"]),
            gr.update(visible=info["show_flux_shift"]),
            gr.update(visible=info["show_max_seq"], value=info["max_seq_default"]),
            gr.update(visible=info["show_hidream_4bit"]),
            gr.update(visible=info["show_min_max_t"], value=info["min_t_default"]),
            gr.update(visible=info["show_min_max_t"], value=info["max_t_default"]),
            gr.update(visible=info["show_llm_lr"]),
            gr.update(visible=info["show_sdxl_lr"]),
            gr.update(value=f"ℹ️ **{info['notes']}**"),
        ]

    with gr.Blocks(title="diffusion-pipe local UI") as app:
        gr.Markdown("# diffusion-pipe local UI")
        gr.Markdown(
            "Local controls for config generation, training, cache jobs, GPU status, and TensorBoard. "
            f"**{len(MODEL_LABELS)} models supported.**"
        )

        with gr.Row():
            config = gr.Dropdown(label="Config", choices=list_configs(), interactive=True)
            refresh = gr.Button("Refresh configs")
            load_btn = gr.Button("📂 Load config")

        with gr.Row():
            gr.Markdown("**VRAM 프리셋** — 모델 선택 후 클릭하면 아래 Config UI 항목에 추천값이 입력됩니다.")

        with gr.Row():
            btn_8gb  = gr.Button("⚡ 8GB  프리셋",  variant="secondary", scale=1)
            btn_16gb = gr.Button("⚡ 16GB 프리셋", variant="primary",   scale=1)
            btn_24gb = gr.Button("⚡ 24GB 프리셋",  variant="secondary", scale=1)
            btn_32gb = gr.Button("⚡ 32GB 프리셋",  variant="secondary", scale=1)

        with gr.Accordion("⚙️ Generate config in UI", open=False) as ui_accordion:
            with gr.Row():
                ui_model = gr.Dropdown(
                    label="Model",
                    choices=list(MODEL_LABELS.keys()),
                    value="Krea 2",
                )
                ui_run_name = gr.Textbox(label="Run name", value="new_lora")
                ui_trigger = gr.Textbox(
                    label="Trigger word / caption prefix",
                    placeholder="Optional, e.g.  je",
                )

            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        ui_dataset = gr.Textbox(
                            label="Dataset folder",
                            placeholder='Example: "C:\\AI\\datasets\\my_dataset"',
                        )
                        btn_dataset = gr.Button("📁", scale=0, min_width=42, variant="secondary")
                with gr.Column():
                    with gr.Row():
                        ui_output = gr.Textbox(
                            label="Output folder",
                            placeholder="Blank = training_runs/<model>/<run_name>",
                        )
                        btn_output = gr.Button("📁", scale=0, min_width=42, variant="secondary")

            with gr.Row():
                ui_main_model = gr.Textbox(
                    label=MODEL_UI["krea2"]["main_label"],
                    placeholder="Path to main model file or folder",
                )
                btn_main_folder = gr.Button("📁", scale=0, min_width=42, variant="secondary")
                btn_main_file   = gr.Button("📄", scale=0, min_width=42, variant="secondary")

            with gr.Row():
                with gr.Column(visible=MODEL_UI["krea2"]["vae_label"] is not None) as ui_vae_col:
                    with gr.Row():
                        ui_vae = gr.Textbox(
                            label=MODEL_UI["krea2"]["vae_label"] or "VAE path",
                            placeholder="VAE .safetensors",
                        )
                        btn_vae = gr.Button("📄", scale=0, min_width=42, variant="secondary")
                with gr.Column(visible=MODEL_UI["krea2"]["te_label"] is not None) as ui_te_col:
                    with gr.Row():
                        ui_te = gr.Textbox(
                            label=MODEL_UI["krea2"]["te_label"] or "Text encoder / LLM path",
                            placeholder="Text encoder .safetensors or folder",
                        )
                        btn_te_folder = gr.Button("📁", scale=0, min_width=42, variant="secondary")
                        btn_te_file   = gr.Button("📄", scale=0, min_width=42, variant="secondary")

            with gr.Row():
                with gr.Column(visible=MODEL_UI["krea2"]["te2_label"] is not None) as ui_te2_col:
                    with gr.Row():
                        ui_te2 = gr.Textbox(
                            label="Secondary text encoder",
                            placeholder="Second TE path (byt5 / clip / etc)",
                        )
                        btn_te2 = gr.Button("📄", scale=0, min_width=42, variant="secondary")
                with gr.Column(visible=MODEL_UI["krea2"]["adapter_label"] is not None) as ui_adapter_col:
                    with gr.Row():
                        ui_adapter = gr.Textbox(
                            label=MODEL_UI["krea2"]["adapter_label"] or "Adapter / extra path",
                            placeholder="Optional path",
                        )
                        btn_adapter = gr.Button("📄", scale=0, min_width=42, variant="secondary")

            ui_model_notes = gr.Markdown(
                value=f"ℹ️ **{MODEL_UI['krea2']['notes']}**"
            )

            with gr.Row():
                ui_resolution = gr.Number(label="Resolution", value=512, precision=0)
                ui_min_ar = gr.Number(label="Min AR", value=0.5)
                ui_max_ar = gr.Number(label="Max AR", value=2.0)
                ui_ar_buckets = gr.Number(label="AR buckets", value=7, precision=0)
                ui_repeats = gr.Number(label="Repeats", value=1, precision=0)

            with gr.Row():
                ui_rank = gr.Number(label="LoRA rank", value=32, precision=0)
                ui_lr = gr.Textbox(label="Learning rate", value="1e-4")
                ui_max_steps = gr.Number(label="Max steps", value=1000, precision=0)
                ui_save_every = gr.Number(label="Save every N steps", value=250, precision=0)

            with gr.Row():
                ui_batch = gr.Number(label="Micro batch", value=1, precision=0)
                ui_grad_accum = gr.Number(label="Gradient accumulation", value=1, precision=0)
                ui_blocks = gr.Number(label="blocks_to_swap", value=16, precision=0)
                ui_cache_batch = gr.Number(label="Caching batch size", value=1, precision=0)

            with gr.Row():
                ui_optimizer = gr.Dropdown(
                    label="Optimizer",
                    choices=["adamw8bitkahan", "adamw_optimi"],
                    value="adamw8bitkahan",
                )
                ui_float8 = gr.Checkbox(label="Use float8 dtype where supported", value=True)
                ui_flux_shift = gr.Checkbox(
                    label="flux_shift = true",
                    value=False,
                    visible=False,
                )
                ui_shift = gr.Number(
                    label="Shift (numeric)",
                    value=3,
                    precision=0,
                    visible=False,
                )
                ui_max_seq = gr.Number(
                    label="Max sequence length",
                    value=768,
                    precision=0,
                    visible=False,
                )

            with gr.Row():
                ui_hidream_4bit = gr.Checkbox(
                    label="HiDream: Llama3 4-bit quantization",
                    value=True,
                    visible=False,
                )
                ui_min_t = gr.Number(
                    label="min_t (Wan2.2 timestep range start)",
                    value=0.0,
                    visible=False,
                )
                ui_max_t = gr.Number(
                    label="max_t (Wan2.2 timestep range end)",
                    value=0.875,
                    visible=False,
                )

            with gr.Row():
                ui_llm_lr = gr.Textbox(
                    label="Anima: llm_adapter_lr",
                    value="0",
                    visible=False,
                )

            with gr.Row(visible=False) as ui_sdxl_lr_row:
                ui_unet_lr = gr.Textbox(label="SDXL unet_lr", value="4e-5")
                ui_te1_lr = gr.Textbox(label="SDXL text_encoder_1_lr", value="2e-5")
                ui_te2_lr = gr.Textbox(label="SDXL text_encoder_2_lr", value="2e-5")

            ui_generate = gr.Button("💾 Generate / Update config", variant="primary")

        with gr.Row():
            action = gr.Radio(
                label="Action",
                choices=["train", "cache_only", "resume latest checkpoint", "resume specific checkpoint"],
                value="train",
            )
            checkpoint = gr.Textbox(
                label="Checkpoint folder",
                placeholder="Only needed for resume specific checkpoint",
            )

        with gr.Row():
            start = gr.Button("▶ Start", variant="primary")
            stop = gr.Button("⏹ Stop job", variant="stop")
            force_stop = gr.Button("⚡ Force stop", variant="stop")
            attach_log = gr.Button("📎 Attach latest UI log")

        with gr.Row():
            log_refresh_interval = gr.Number(
                label="로그 자동 갱신 주기 (초, 0 = 수동)",
                value=3, precision=0, minimum=0, maximum=60, scale=1,
            )
            log_refresh_btn = gr.Button("🔄 로그 수동 갱신", scale=2)

        status = gr.Textbox(label="Status", lines=4)
        log = gr.Textbox(label="Training log tail (실시간)", lines=18)
        log_timer = gr.Timer(value=3, active=False)

        with gr.Row():
            gpu = gr.Button("📊 GPU / process status")
            runs = gr.Button("📁 Recent run files")
            checkpoints = gr.Button("💾 Saved checkpoints")

        info = gr.Textbox(label="Info", lines=12)

        with gr.Row():
            tb_logdir = gr.Textbox(label="TensorBoard logdir", value=str(RUN_ROOT))
            tb_port = gr.Textbox(label="Port", value="6006")

        with gr.Row():
            tb_start = gr.Button("Start TensorBoard")
            tb_stop = gr.Button("Stop TensorBoard")

        tb_status = gr.Textbox(label="TensorBoard status", lines=2)
        tb_embed = gr.HTML(label="TensorBoard")

        # --- event wiring ---

        dynamic_outputs = [
            ui_main_model,
            ui_vae_col, ui_vae,
            ui_te_col,  ui_te,
            ui_te2_col, ui_te2,
            ui_adapter_col, ui_adapter,
            ui_shift,
            ui_flux_shift,
            ui_max_seq,
            ui_hidream_4bit,
            ui_min_t,
            ui_max_t,
            ui_llm_lr,
            ui_sdxl_lr_row,
            ui_model_notes,
        ]

        ui_model.change(update_model_ui, inputs=ui_model, outputs=dynamic_outputs)

        _load_outputs = [
            status, ui_accordion,
            ui_model, ui_run_name, ui_trigger, ui_dataset, ui_output,
            ui_main_model,
            ui_vae_col, ui_vae,
            ui_te_col,  ui_te,
            ui_te2_col, ui_te2,
            ui_adapter_col, ui_adapter,
            ui_model_notes,
            ui_resolution, ui_min_ar, ui_max_ar, ui_ar_buckets, ui_repeats,
            ui_rank, ui_lr, ui_max_steps, ui_save_every,
            ui_batch, ui_grad_accum, ui_blocks, ui_cache_batch,
            ui_optimizer, ui_float8, ui_flux_shift, ui_shift, ui_max_seq,
            ui_hidream_4bit, ui_min_t, ui_max_t,
            ui_llm_lr, ui_sdxl_lr_row, ui_unet_lr, ui_te1_lr, ui_te2_lr,
        ]
        load_btn.click(load_config_to_ui, inputs=config, outputs=_load_outputs)

        # --- path browse buttons ---
        btn_dataset.click(browse_folder,  inputs=ui_dataset, outputs=ui_dataset)
        btn_output.click( browse_folder,  inputs=ui_output,  outputs=ui_output)
        btn_main_folder.click(browse_folder, inputs=ui_main_model, outputs=ui_main_model)
        btn_main_file.click(
            lambda x: browse_file(x, _SF_GGUF), inputs=ui_main_model, outputs=ui_main_model
        )
        btn_vae.click(
            lambda x: browse_file(x, _SF),       inputs=ui_vae,  outputs=ui_vae
        )
        btn_te_folder.click(browse_folder,        inputs=ui_te,   outputs=ui_te)
        btn_te_file.click(
            lambda x: browse_file(x, _SF_GGUF),  inputs=ui_te,   outputs=ui_te
        )
        btn_te2.click(
            lambda x: browse_file(x, _SF),        inputs=ui_te2,  outputs=ui_te2
        )
        btn_adapter.click(
            lambda x: browse_file(x, _SF),        inputs=ui_adapter, outputs=ui_adapter
        )

        _preset_outputs = [
            status, ui_resolution, ui_rank, ui_lr,
            ui_blocks, ui_float8, ui_batch, ui_grad_accum, ui_max_seq,
        ]
        btn_8gb.click( lambda m: apply_vram_preset(m,  8), inputs=ui_model, outputs=_preset_outputs)
        btn_16gb.click(lambda m: apply_vram_preset(m, 16), inputs=ui_model, outputs=_preset_outputs)
        btn_24gb.click(lambda m: apply_vram_preset(m, 24), inputs=ui_model, outputs=_preset_outputs)
        btn_32gb.click(lambda m: apply_vram_preset(m, 32), inputs=ui_model, outputs=_preset_outputs)

        ui_generate.click(
            generate_config_from_ui,
            inputs=[
                ui_model, ui_run_name, ui_dataset, ui_output, ui_trigger,
                ui_main_model, ui_vae, ui_te, ui_te2, ui_adapter,
                ui_resolution, ui_min_ar, ui_max_ar, ui_ar_buckets, ui_repeats,
                ui_rank, ui_lr, ui_max_steps, ui_save_every,
                ui_batch, ui_grad_accum, ui_blocks, ui_cache_batch,
                ui_optimizer, ui_float8, ui_flux_shift, ui_shift, ui_max_seq,
                ui_llm_lr, ui_unet_lr, ui_te1_lr, ui_te2_lr,
                ui_hidream_4bit, ui_min_t, ui_max_t,
            ],
            outputs=[status, config],
        )

        def _tail_log():
            return tail(TRAIN_LOG)

        def _start_with_timer(cfg, act, ckpt, interval):
            st, lg = start_training(cfg, act, ckpt)
            active = int(interval or 0) > 0
            iv = max(1, int(interval or 3))
            return st, lg, gr.update(active=active, value=iv)

        def _stop_with_timer():
            st, lg = stop_training()
            return st, lg, gr.update(active=False)

        def _force_stop_with_timer():
            st, lg = force_stop_training()
            return st, lg, gr.update(active=False)

        def _set_timer_interval(interval):
            iv = int(interval or 0)
            if iv > 0:
                return gr.update(active=True, value=iv)
            return gr.update(active=False)

        refresh.click(refresh_configs, outputs=config)
        start.click(
            _start_with_timer,
            inputs=[config, action, checkpoint, log_refresh_interval],
            outputs=[status, log, log_timer],
        )
        stop.click(_stop_with_timer, outputs=[status, log, log_timer])
        force_stop.click(_force_stop_with_timer, outputs=[status, log, log_timer])
        attach_log.click(attach_latest_log, outputs=[status, log])
        log_refresh_btn.click(_tail_log, outputs=log)
        log_refresh_interval.change(_set_timer_interval, inputs=log_refresh_interval, outputs=log_timer)
        log_timer.tick(_tail_log, outputs=log)
        gpu.click(gpu_status, outputs=info)
        runs.click(recent_runs, outputs=info)
        checkpoints.click(saved_checkpoints, inputs=config, outputs=info)
        tb_start.click(start_tensorboard, inputs=[tb_logdir, tb_port], outputs=[tb_status, tb_embed])
        tb_stop.click(stop_tensorboard, outputs=[tb_status, tb_embed])

    return app


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args, _ = parser.parse_known_args()
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=args.port, inbrowser=False)


if __name__ == "__main__":
    main()
