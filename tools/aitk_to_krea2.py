#!/usr/bin/env python3
import re
from pathlib import Path

import yaml


REPO_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_DIR / "configs" / "generated" / "krea2"
DEFAULT_OUTPUT_ROOT = (REPO_DIR / "training_runs" / "krea2").as_posix()


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
        value = ask(prompt, str(default))
        try:
            return int(value)
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
    value = ask("Retry or cancel? (r/c)", "r").lower()
    if value in ("c", "cancel", "q", "quit"):
        raise UserCancelled()


def require_file(label: str, default: str) -> str:
    while True:
        path = win_to_wsl(ask(label, default))
        if Path(path).is_file():
            return path
        print(f"Missing required file: {path}")
        retry_or_cancel()


def require_dir(label: str, default: str) -> str:
    while True:
        path = win_to_wsl(ask(label, default))
        if Path(path).is_dir():
            return path
        print(f"Missing required folder: {path}")
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


def list_aitk_configs() -> list[Path]:
    roots = [Path("/mnt/c/ai/AI-Toolkit/output"), Path("/mnt/c/ai/AI-Toolkit/config")]
    configs: list[Path] = []
    for root in roots:
        if root.exists():
            configs.extend(root.rglob("config.yaml"))
            configs.extend(root.rglob("*.yaml"))
    unique = []
    seen = set()
    for path in configs:
        key = path.as_posix()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return sorted(unique)


def choose_yaml() -> Path:
    configs = list_aitk_configs()
    print("AI-Toolkit config.yaml -> diffusion-pipe Krea2 converter")
    if configs:
        print("")
        for idx, path in enumerate(configs[:40], start=1):
            print(f"  {idx}. {path}")
        if len(configs) > 40:
            print(f"  ... {len(configs) - 40} more omitted. Enter a path manually if needed.")

    while True:
        default = "1" if configs else ""
        value = ask("\nSelect AI-Toolkit config number or enter YAML path", default)
        if value.isdigit() and configs:
            idx = int(value)
            if 1 <= idx <= min(len(configs), 40):
                return configs[idx - 1]
            print("Invalid config number.")
            continue
        path = Path(win_to_wsl(value))
        if path.is_file():
            return path
        print(f"Missing YAML file: {path}")
        retry_or_cancel()


def first_process(data: dict) -> dict:
    process = data.get("config", {}).get("process", [])
    if not process:
        raise RuntimeError("Could not find config.process[0] in AI-Toolkit YAML.")
    return process[0]


def as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def convert() -> tuple[Path, Path]:
    yaml_path = choose_yaml()
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    proc = first_process(data)

    name = data.get("config", {}).get("name") or data.get("meta", {}).get("name") or yaml_path.parent.name
    train = proc.get("train", {})
    network = proc.get("network", {})
    save = proc.get("save", {})
    datasets = proc.get("datasets", [])
    if not datasets:
        raise RuntimeError("Could not find process.datasets in AI-Toolkit YAML.")

    model = proc.get("model", {})
    arch = str(model.get("arch", ""))
    if "krea2" not in arch.lower() and "krea" not in str(model.get("name_or_path", "")).lower():
        print("Warning: this AI-Toolkit config does not look like Krea2.")
        if not ask_bool("Convert anyway", False):
            raise UserCancelled()

    run_name = ask("Run name", name)
    output_dir = win_to_wsl(ask("Output folder", f"{DEFAULT_OUTPUT_ROOT}/{run_name}"))
    trigger_word = ask("Trigger word / caption prefix (blank to skip)", str(proc.get("trigger_word", "") or ""))
    caption_prefix = f"{trigger_word.strip()}, " if trigger_word.strip() else ""

    diffusion_model = require_file(
        "Krea2 base diffusion model",
        "/mnt/m/Models/diffusion_models/Krea2/krea2_turbo_mxfp8.safetensors",
    )
    vae = require_file("Qwen Image VAE", find_vae("qwen_image_vae.safetensors"))
    text_encoder = require_file(
        "Qwen3VL text encoder",
        find_text_encoder("qwen_3vl_4b"),
    )

    rank = int(network.get("linear", network.get("rank", 16)))
    max_steps = int(train.get("steps", 1000))
    grad_accum = int(train.get("gradient_accumulation", 1))
    lr = train.get("lr", "1e-4")
    save_every = int(save.get("save_every", 250))
    batch_size = int(train.get("batch_size", 1))
    gradient_checkpointing = bool(train.get("gradient_checkpointing", True))
    weight_decay = train.get("optimizer_params", {}).get("weight_decay", 0.01)
    optimizer_name = str(train.get("optimizer", "adamw8bit")).lower()
    blocks_to_swap = ask_int("blocks_to_swap, 0 preserves AI-Toolkit-style no offload", 0)
    cache_batch = ask_int("Caching batch size", 1)

    dataset_config = CONFIG_DIR / f"{run_name}.dataset.toml"
    train_config = CONFIG_DIR / f"{run_name}.toml"

    dataset_lines = []
    first_res = as_list(datasets[0].get("resolution", [512]))
    dataset_lines.append(f"resolutions = [{', '.join(str(int(x)) for x in first_res)}]")
    dataset_lines.append("enable_ar_bucket = true")
    dataset_lines.append("min_ar = 0.5")
    dataset_lines.append("max_ar = 2.0")
    dataset_lines.append("num_ar_buckets = 7")
    dataset_lines.append("frame_buckets = [1]")
    dataset_lines.append("skip_empty_caption = false")
    dataset_lines.append("")

    for ds in datasets:
        folder = require_dir("Dataset folder from AI-Toolkit config", ds.get("folder_path", ""))
        repeats = int(ds.get("num_repeats", 1))
        mask = ds.get("mask_path")
        dataset_lines.append("[[directory]]")
        dataset_lines.append(f"path = {quote_toml(folder)}")
        dataset_lines.append(f"num_repeats = {repeats}")
        dataset_lines.append(f"caption_prefix = {quote_toml(caption_prefix)}")
        if mask:
            mask_path = win_to_wsl(str(mask))
            if Path(mask_path).is_dir():
                dataset_lines.append(f"mask_path = {quote_toml(mask_path)}")
            else:
                print(f"Warning: mask_path ignored because it does not exist: {mask_path}")
        dataset_lines.append("")

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
activation_checkpointing = {str(gradient_checkpointing).lower()}
partition_method = 'parameters'
save_dtype = 'bfloat16'
caching_batch_size = {cache_batch}
steps_per_print = 1
{blocks_line}
[model]
type = 'krea2'
diffusion_model = {quote_toml(diffusion_model)}
vae = {quote_toml(vae)}
text_encoders = [
    {{path = {quote_toml(text_encoder)}, type = 'krea2'}}
]
dtype = 'bfloat16'
diffusion_model_dtype = 'float8'
timestep_sample_method = 'logit_normal'

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
    dataset_config.parent.mkdir(parents=True, exist_ok=True)
    dataset_config.write_text("\n".join(dataset_lines), encoding="utf-8", newline="\n")
    train_config.write_text(train_text, encoding="utf-8", newline="\n")
    return dataset_config, train_config


def main() -> int:
    dataset_config, train_config = convert()
    print("")
    print("Converted config")
    print(f"  dataset config: {dataset_config}")
    print(f"  train config:   {train_config}")
    print("")
    print("Run later with:")
    print("  C:\\ai\\diffusion-pipe\\krea2_run.bat")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UserCancelled:
        print("\nCancelled. No config was written.")
        raise SystemExit(2)
