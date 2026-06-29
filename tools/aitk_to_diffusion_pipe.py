#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import yaml


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
        return f"/mnt/{match.group(1).lower()}/{match.group(2).replace('\\', '/')}"
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


def list_aitk_configs() -> list[Path]:
    roots = [Path("/mnt/c/ai/AI-Toolkit/output"), Path("/mnt/c/ai/AI-Toolkit/config")]
    found = []
    for root in roots:
        if root.exists():
            found.extend(root.rglob("config.yaml"))
    return sorted(dict.fromkeys(found))


def choose_yaml(default_config: str | None = None) -> Path:
    if default_config:
        path = Path(win_to_wsl(default_config))
        if path.is_file():
            return path
        print(f"Provided AI-Toolkit config.yaml was not found: {path}")

    configs = list_aitk_configs()
    print("AI-Toolkit config.yaml -> diffusion-pipe converter")
    print("AI-Toolkit does not need to be installed. You only need an existing config.yaml.")
    for idx, path in enumerate(configs[:50], start=1):
        print(f"  {idx}. {path}")
    while True:
        value = ask("\nSelect AI-Toolkit config number or enter YAML path", "1" if configs else "")
        if value.isdigit() and configs:
            idx = int(value)
            if 1 <= idx <= min(len(configs), 50):
                return configs[idx - 1]
        path = Path(win_to_wsl(value))
        if path.is_file():
            return path
        print(f"Missing YAML file: {path}")
        retry_or_cancel()


def choose_model() -> tuple[str, str]:
    print("\nTarget model")
    for key, (_, label) in MODELS.items():
        print(f"  {key}. {label}")
    while True:
        value = ask("Select model", "1")
        if value in MODELS:
            return MODELS[value]
        print("Please select a listed model number.")


def first_process(data: dict) -> dict:
    process = data.get("config", {}).get("process", [])
    if not process:
        raise RuntimeError("Could not find config.process[0] in AI-Toolkit YAML.")
    return process[0]


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


def convert(default_config: str | None = None) -> tuple[Path, Path]:
    yaml_path = choose_yaml(default_config)
    model_key, model_label = choose_model()
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    proc = first_process(data)
    train = proc.get("train", {})
    network = proc.get("network", {})
    save = proc.get("save", {})
    datasets = proc.get("datasets", [])
    if not datasets:
        raise RuntimeError("Could not find process.datasets in AI-Toolkit YAML.")

    source_name = data.get("config", {}).get("name") or data.get("meta", {}).get("name") or yaml_path.parent.name
    run_name = ask("Run name", f"{source_name}_{model_key}")
    output_dir = win_to_wsl(ask("Output folder", f"{OUTPUT_ROOT}/{model_key}/{run_name}"))
    trigger_word = ask("Trigger word / caption prefix (blank to skip)", str(proc.get("trigger_word", "") or ""))
    caption_prefix = f"{trigger_word.strip()}, " if trigger_word.strip() else ""

    rank = int(network.get("linear", network.get("rank", 16)))
    max_steps = int(train.get("steps", 1000))
    grad_accum = int(train.get("gradient_accumulation", 1))
    lr = train.get("lr", "1e-4")
    save_every = int(save.get("save_every", 250))
    batch_size = int(train.get("batch_size", 1))
    gradient_checkpointing = bool(train.get("gradient_checkpointing", True))
    weight_decay = train.get("optimizer_params", {}).get("weight_decay", 0.0001)
    optimizer_name = str(train.get("optimizer", "adamw8bit")).lower()

    rank = ask_int("LoRA rank", rank)
    max_steps = ask_int("Max steps", max_steps)
    lr = ask("Learning rate", str(lr))
    save_every = ask_int("Save every N steps", save_every)
    grad_accum = ask_int("Gradient accumulation steps", grad_accum)
    blocks_to_swap = ask_int("blocks_to_swap", 0)
    cache_batch = ask_int("Caching batch size", 1)

    first_res = datasets[0].get("resolution", [512])
    first_res = first_res if isinstance(first_res, list) else [first_res]
    dataset_lines = [
        f"resolutions = [{', '.join(str(int(x)) for x in first_res)}]",
        "enable_ar_bucket = true",
        "min_ar = 0.5",
        "max_ar = 2.0",
        "num_ar_buckets = 7",
        "frame_buckets = [1]",
        "skip_empty_caption = false",
        "",
    ]
    for ds in datasets:
        folder = require_path("Dataset folder from AI-Toolkit config", str(ds.get("folder_path", "")), "dir")
        dataset_lines.extend([
            "[[directory]]",
            f"path = {quote_toml(folder)}",
            f"num_repeats = {int(ds.get('num_repeats', 1))}",
            f"caption_prefix = {quote_toml(caption_prefix)}",
            "",
        ])

    if "8bit" in optimizer_name:
        optimizer_text = f"""[optimizer]
type = 'AdamW8bitKahan'
lr = {lr}
betas = [0.9, 0.99]
weight_decay = {weight_decay}
stabilize = false
"""
    else:
        optimizer_text = f"""[optimizer]
type = 'adamw_optimi'
lr = {lr}
betas = [0.9, 0.99]
weight_decay = {weight_decay}
eps = 1e-8
"""

    model_text = model_block(model_key)
    blocks_line = f"blocks_to_swap = {blocks_to_swap}\n" if blocks_to_swap > 0 else ""
    train_text = f"""output_dir = {quote_toml(output_dir)}
dataset = {quote_toml((CONFIG_ROOT / model_key / f'{run_name}.dataset.toml').as_posix())}

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
activation_checkpointing = {str(gradient_checkpointing).lower()}
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

    config_dir = CONFIG_ROOT / model_key
    dataset_config = config_dir / f"{run_name}.dataset.toml"
    train_config = config_dir / f"{run_name}.toml"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    dataset_config.write_text("\n".join(dataset_lines), encoding="utf-8", newline="\n")
    train_config.write_text(train_text, encoding="utf-8", newline="\n")
    return dataset_config, train_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert an AI-Toolkit config.yaml to diffusion-pipe configs.")
    parser.add_argument("--config", help="Path to an AI-Toolkit config.yaml. Windows paths and quoted paths are accepted.")
    args = parser.parse_args()

    dataset_config, train_config = convert(args.config)
    print("\nConverted config")
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
