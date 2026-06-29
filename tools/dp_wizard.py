#!/usr/bin/env python3
import re
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_DIR / "configs" / "generated"
OUTPUT_ROOT = (REPO_DIR / "training_runs").as_posix()


MODELS = {
    "1": ("krea2", "Krea 2"),
    "2": ("z_image", "Z-Image"),
    "3": ("flux2_klein9b", "Flux 2 Klein 9B"),
    "4": ("ltx2", "LTX 2.3"),
    "5": ("anima", "Anima"),
    "6": ("sd3", "Stable Diffusion 3"),
    "7": ("sdxl", "SDXL"),
    "8": ("auraflow", "AuraFlow"),
}


class UserCancelled(Exception):
    pass


def win_to_wsl(value: str) -> str:
    value = value.strip()
    while len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1].strip()
    value = value.strip('"').strip("'").strip()
    match = re.match(r"^([a-zA-Z]):[\\/](.*)$", value)
    if match:
        drive = match.group(1).lower()
        rest = match.group(2).replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return value.replace("\\", "/")


def quote_toml(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default not in (None, "") else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return default if value == "" and default is not None else value


def ask_int(prompt: str, default: int) -> int:
    while True:
        try:
            return int(ask(prompt, str(default)))
        except ValueError:
            print("Please enter a number.")


def ask_float(prompt: str, default: float) -> float:
    while True:
        try:
            return float(ask(prompt, str(default)))
        except ValueError:
            print("Please enter a number.")


def ask_bool(prompt: str, default: bool) -> bool:
    default_text = "y" if default else "n"
    while True:
        value = ask(prompt + " (y/n)", default_text).lower()
        if value in ("y", "yes", "1", "true"):
            return True
        if value in ("n", "no", "0", "false"):
            return False
        print("Please enter y or n.")


def retry_or_cancel() -> None:
    if ask("Retry or cancel? (r/c)", "r").lower() in ("c", "cancel", "q", "quit"):
        raise UserCancelled()


def require_path(label: str, default: str, kind: str = "file") -> str:
    while True:
        path = Path(win_to_wsl(ask(label, default)))
        ok = path.is_dir() if kind == "dir" else path.is_file()
        if ok:
            return path.as_posix()
        print(f"Missing required {kind}: {path}")
        retry_or_cancel()


def find_vae(*names: str) -> str:
    roots = [Path("/mnt/c/AI/models/vae"), Path("/mnt/c/ai/models/vae")]
    for root in roots:
        for name in names:
            path = root / name
            if path.is_file():
                return path.as_posix()
    return f"/mnt/c/AI/models/vae/{names[0]}"


def find_text_encoder(*patterns: str) -> str:
    roots = [Path("/mnt/m/Models/text_encoders"), Path("/mnt/c/AI/models/text_encoders")]
    lowered = [p.lower() for p in patterns]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in (".safetensors", ".gguf"):
                haystack = path.as_posix().lower()
                if all(part in haystack for part in lowered):
                    return path.as_posix()
    joined = "_".join(patterns)
    return f"/mnt/m/Models/text_encoders/{joined}.safetensors"


def optional_file(label: str, default: str = "") -> str:
    value = ask(label + " (blank to skip)", default)
    if not value:
        return ""
    path = Path(win_to_wsl(value))
    if path.is_file():
        return path.as_posix()
    print(f"Missing optional file, skipped: {path}")
    return ""


def choose_model() -> tuple[str, str]:
    print("diffusion-pipe multi-model wizard")
    print("\nTarget model")
    for key, (_, label) in MODELS.items():
        print(f"  {key}. {label}")
    while True:
        value = ask("Select model", "1")
        if value in MODELS:
            return MODELS[value]
        print("Please select a listed model number.")


def model_defaults(model_key: str) -> dict:
    defaults = {
        "resolution": 512,
        "rank": 16,
        "lr": "1e-4",
        "max_steps": 1000,
        "save_every": 250,
        "grad_accum": 1,
        "blocks": 0,
    }
    if model_key in ("krea2", "z_image", "flux2_klein9b"):
        defaults.update({"rank": 32, "grad_accum": 4 if model_key == "krea2" else 1})
    if model_key == "ltx2":
        defaults.update({"rank": 16, "resolution": 512, "blocks": 46})
    if model_key == "sdxl":
        defaults.update({"lr": "1e-4", "rank": 16})
    if model_key == "anima":
        defaults.update({"lr": "5e-5", "rank": 16})
    return defaults


def model_block(model_key: str) -> str:
    if model_key == "krea2":
        dm = require_path("Krea2 diffusion model", "/mnt/m/Models/diffusion_models/Krea2/krea2_turbo_mxfp8.safetensors")
        vae = require_path("Qwen Image VAE", find_vae("qwen_image_vae.safetensors"))
        te = require_path("Qwen3VL 4B text encoder", find_text_encoder("qwen_3vl_4b"))
        return f"""[model]
type = 'krea2'
diffusion_model = {quote_toml(dm)}
vae = {quote_toml(vae)}
text_encoders = [
    {{path = {quote_toml(te)}, type = 'krea2'}}
]
dtype = 'bfloat16'
diffusion_model_dtype = 'float8'
timestep_sample_method = 'logit_normal'
"""

    if model_key == "z_image":
        dm = require_path("Z-Image diffusion model", "/mnt/m/Models/diffusion_models/Z-Image/z_image_turbo_bf16.safetensors")
        vae = require_path("Flux VAE", find_vae("z-Image-Vae.safetensors", "flux1-vae.safetensors", "ae.safetensors"))
        te = require_path("Qwen 3 4B text encoder", find_text_encoder("qwen_3_4b.safetensors"))
        adapter = optional_file("Z-Image Turbo training adapter", "/mnt/m/Models/loras/zimage_turbo_training_adapter_v1.safetensors")
        merge = f"merge_adapters = [{quote_toml(adapter)}]\n" if adapter else ""
        dtype_line = "diffusion_model_dtype = 'float8'\n" if ask_bool("Use float8 diffusion_model_dtype", True) else ""
        return f"""[model]
type = 'z_image'
diffusion_model = {quote_toml(dm)}
vae = {quote_toml(vae)}
text_encoders = [
    {{path = {quote_toml(te)}, type = 'lumina2'}}
]
{merge}dtype = 'bfloat16'
{dtype_line}"""

    if model_key == "flux2_klein9b":
        dm = require_path("Flux 2 Klein 9B base diffusion model", "/mnt/m/Models/diffusion_models/Flux2/flux-2-klein-base-9b.safetensors")
        vae = require_path("Flux2 VAE", find_vae("flux2-vae.safetensors"))
        te = require_path("Qwen 3 8B text encoder", find_text_encoder("qwen", "klein9b"))
        return f"""[model]
type = 'flux2'
diffusion_model = {quote_toml(dm)}
vae = {quote_toml(vae)}
text_encoders = [
    {{path = {quote_toml(te)}, type = 'flux2'}}
]
dtype = 'bfloat16'
diffusion_model_dtype = 'float8'
timestep_sample_method = 'logit_normal'
shift = 3
"""

    if model_key == "ltx2":
        dm = require_path("LTX 2.3 diffusion model", "/mnt/m/Models/diffusion_models/LTX/ltx-2.3-22b-dev.safetensors")
        te = require_path("Gemma 3 12B text encoder", find_text_encoder("gemma_3_12b", "fp4_mixed"))
        shift = ask_int("Shift", 1)
        return f"""[model]
type = 'ltx2'
diffusion_model = {quote_toml(dm)}
text_encoder = {quote_toml(te)}
dtype = 'bfloat16'
diffusion_model_dtype = 'float8'
timestep_sample_method = 'logit_normal'
shift = {shift}
"""

    if model_key == "anima":
        tr = require_path("Anima transformer", "/mnt/m/Models/diffusion_models/Anima/anima-preview.safetensors")
        vae = require_path("Qwen Image VAE", find_vae("qwen_image_vae.safetensors"))
        llm = require_path("Qwen3 0.6B base", find_text_encoder("qwen", "06b"))
        llm_lr = ask("llm_adapter_lr", "0")
        return f"""[model]
type = 'anima'
transformer_path = {quote_toml(tr)}
vae_path = {quote_toml(vae)}
llm_path = {quote_toml(llm)}
dtype = 'bfloat16'
llm_adapter_lr = {llm_lr}
"""

    if model_key == "sd3":
        diff = require_path("SD3/SD3.5 Diffusers folder", "/mnt/m/Models/diffusers/stable-diffusion-3.5-medium", "dir")
        fp8 = "transformer_dtype = 'float8'\n" if ask_bool("Use float8 transformer_dtype", True) else ""
        return f"""[model]
type = 'sd3'
diffusers_path = {quote_toml(diff)}
dtype = 'bfloat16'
{fp8}"""

    if model_key == "sdxl":
        ckpt = require_path("SDXL checkpoint safetensors", "/mnt/m/Models/checkpoints/sd_xl_base_1.0.safetensors")
        unet_lr = ask("SDXL unet_lr", "4e-5")
        te1_lr = ask("SDXL text_encoder_1_lr", "2e-5")
        te2_lr = ask("SDXL text_encoder_2_lr", "2e-5")
        return f"""[model]
type = 'sdxl'
checkpoint_path = {quote_toml(ckpt)}
dtype = 'bfloat16'
unet_lr = {unet_lr}
text_encoder_1_lr = {te1_lr}
text_encoder_2_lr = {te2_lr}
"""

    if model_key == "auraflow":
        tr = require_path("AuraFlow transformer", "/mnt/m/Models/diffusion_models/AuraFlow/pony-v7-base.safetensors")
        te = require_path("AuraFlow UMT5 text encoder", find_text_encoder("umt5", "auraflow"))
        vae = require_path("SDXL VAE", find_vae("sdxl_vae.safetensors"))
        max_len = ask_int("max_sequence_length, 768 for Pony-V7, 256 for base AuraFlow", 768)
        return f"""[model]
type = 'auraflow'
transformer_path = {quote_toml(tr)}
text_encoder_path = {quote_toml(te)}
vae_path = {quote_toml(vae)}
dtype = 'bfloat16'
transformer_dtype = 'float8'
timestep_sample_method = 'logit_normal'
max_sequence_length = {max_len}
"""

    raise RuntimeError(f"Unhandled model: {model_key}")


def main() -> int:
    model_key, model_label = choose_model()
    defaults = model_defaults(model_key)

    run_name = ask("\nRun name", f"{model_key}_test")
    dataset_path = require_path("Training image/video folder", "/mnt/c/ai/input", "dir")
    output_dir = win_to_wsl(ask("Output folder", f"{OUTPUT_ROOT}/{model_key}/{run_name}"))

    resolution = ask_int("\nResolution", defaults["resolution"])
    min_ar = ask_float("Min aspect ratio", 0.5)
    max_ar = ask_float("Max aspect ratio", 2.0)
    num_ar_buckets = ask_int("AR bucket count", 7)
    repeats = ask_int("Dataset repeats, num_repeats", 1)
    trigger_word = ask("Trigger word / caption prefix (blank to skip)", "")
    caption_prefix = f"{trigger_word.strip()}, " if trigger_word.strip() else ""

    rank = ask_int("\nLoRA rank", defaults["rank"])
    lr = ask("Learning rate", defaults["lr"])
    max_steps = ask_int("Max steps", defaults["max_steps"])
    save_every = ask_int("Save every N steps", defaults["save_every"])
    batch_size = ask_int("Micro batch size per GPU", 1)
    grad_accum = ask_int("Gradient accumulation steps", defaults["grad_accum"])
    blocks_to_swap = ask_int("blocks_to_swap", defaults["blocks"])
    cache_batch = ask_int("Caching batch size", 1)
    optimizer = ask("Optimizer: adamw8bitkahan or adamw_optimi", "adamw8bitkahan").lower()
    if optimizer not in ("adamw8bitkahan", "adamw_optimi"):
        optimizer = "adamw8bitkahan"

    model_text = model_block(model_key)

    config_dir = CONFIG_ROOT / model_key
    dataset_config = config_dir / f"{run_name}.dataset.toml"
    train_config = config_dir / f"{run_name}.toml"

    dataset_text = f"""resolutions = [{resolution}]
enable_ar_bucket = true
min_ar = {min_ar}
max_ar = {max_ar}
num_ar_buckets = {num_ar_buckets}
frame_buckets = [1]
skip_empty_caption = false

[[directory]]
path = {quote_toml(dataset_path)}
num_repeats = {repeats}
caption_prefix = {quote_toml(caption_prefix)}
"""

    if optimizer == "adamw_optimi":
        optimizer_text = f"""[optimizer]
type = 'adamw_optimi'
lr = {lr}
betas = [0.9, 0.99]
weight_decay = 0.0001
eps = 1e-8
"""
    else:
        optimizer_text = f"""[optimizer]
type = 'AdamW8bitKahan'
lr = {lr}
betas = [0.9, 0.99]
weight_decay = 0.0001
stabilize = false
"""

    blocks_line = f"blocks_to_swap = {blocks_to_swap}\n" if blocks_to_swap > 0 else ""
    train_text = f"""output_dir = {quote_toml(output_dir)}
dataset = {quote_toml(dataset_config.as_posix())}

epochs = 1000
max_steps = {max_steps}
micro_batch_size_per_gpu = {batch_size}
pipeline_stages = 1
gradient_accumulation_steps = {grad_accum}
gradient_clipping = 1.0
warmup_steps = 100
eval_before_first_step = false
eval_every_n_steps = {max(save_every, 1)}
eval_micro_batch_size_per_gpu = 1
eval_gradient_accumulation_steps = 1
save_every_n_steps = {save_every}
checkpoint_every_n_minutes = 120
activation_checkpointing = true
partition_method = 'parameters'
save_dtype = 'bfloat16'
caching_batch_size = {cache_batch}
steps_per_print = 1
{blocks_line}
{model_text}
[adapter]
type = 'lora'
rank = {rank}
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

    print("\nGenerated configs")
    print(f"  model:          {model_label}")
    print(f"  dataset config: {dataset_config}")
    print(f"  train config:   {train_config}")
    print("\nRun later with:")
    print("  C:\\ai\\diffusion-pipe\\dp_run.bat")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UserCancelled:
        print("\nCancelled. No config was written.")
        raise SystemExit(2)
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled. No config was written.")
        raise SystemExit(130)
