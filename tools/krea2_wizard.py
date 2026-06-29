#!/usr/bin/env python3
import re
import shlex
import subprocess
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_DIR / "configs" / "generated" / "krea2"
DEFAULT_OUTPUT_DIR = (REPO_DIR / "training_runs" / "krea2").as_posix()


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


def ask_float(prompt: str, default: float) -> float:
    while True:
        value = ask(prompt, str(default))
        try:
            return float(value)
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


def warn_before_gpu_work(action: str) -> bool:
    print("")
    print(f"About to run {action}.")
    print("Do this only when AI-Toolkit/ComfyUI/other training jobs are not using the GPU.")
    print("If another job is running, choose n and run this later.")
    return ask_bool(f"Run {action} now", False)


def ask_retry_or_cancel() -> None:
    while True:
        value = ask("Retry path input or cancel? (r/c)", "r").lower()
        if value in ("r", "retry"):
            return
        if value in ("c", "cancel", "q", "quit"):
            raise UserCancelled()
        print("Please enter r or c.")


def find_candidates(
    patterns: list[str],
    roots: list[Path],
    limit: int = 12,
    exclude_parts: list[str] | None = None,
    suffixes: tuple[str, ...] = (".safetensors",),
) -> list[str]:
    found: list[str] = []
    lowered = [p.lower() for p in patterns]
    excluded = [p.lower() for p in (exclude_parts or [])]
    for root in roots:
        if not root.exists():
            continue
        try:
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                full = path.as_posix().lower()
                name = path.name.lower()
                if suffixes and path.suffix.lower() not in suffixes:
                    continue
                if any(part in full for part in excluded):
                    continue
                if all(part in name for part in lowered):
                    found.append(path.as_posix())
                    if len(found) >= limit:
                        return found
        except PermissionError:
            continue
    return found


def require_existing_file(label: str, path_text: str) -> str:
    path = win_to_wsl(path_text)
    if Path(path).is_file():
        return path
    print(f"Missing required file for {label}: {path}")
    ask_retry_or_cancel()
    return ""


def require_existing_dir(label: str, path_text: str) -> str:
    path = win_to_wsl(path_text)
    if Path(path).is_dir():
        return path
    print(f"Missing required folder for {label}: {path}")
    ask_retry_or_cancel()
    return ""


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


def choose_file(label: str, patterns: list[str], default: str, exclude_parts: list[str] | None = None) -> str:
    roots = [
        Path("/mnt/c/ai/models"),
        Path("/mnt/c/ai/ComfyUI/models"),
        Path("/mnt/c/ai/ComfyUI-Easy-Install/ComfyUI/models"),
        Path("/mnt/c/ai/Comfy-Dasktop/ComfyUI/models"),
    ]
    candidates = find_candidates(patterns, roots, exclude_parts=exclude_parts)
    while True:
        print(f"\n[{label}]")
        if candidates:
            for idx, path in enumerate(candidates, start=1):
                print(f"  {idx}. {path}")
        else:
            print("  No local candidate found. Enter the file path manually.")

        value = ask("Select number or enter path", default)
        if value.isdigit() and candidates:
            idx = int(value)
            if 1 <= idx <= len(candidates):
                value = candidates[idx - 1]
            else:
                print("Invalid candidate number.")
                continue

        checked = require_existing_file(label, value)
        if checked:
            return checked


def choose_dir(label: str, default: str) -> str:
    while True:
        value = ask(label, default)
        checked = require_existing_dir(label, value)
        if checked:
            return checked


def choose_output_dir(default: str) -> str:
    path = win_to_wsl(ask("Output folder", default))
    return path


def write_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def build_configs(
    run_name: str,
    dataset_path: str,
    output_dir: str,
    diffusion_model: str,
    vae: str,
    text_encoder: str,
    resolution: int,
    min_ar: float,
    max_ar: float,
    num_ar_buckets: int,
    repeats: int,
    caption_prefix: str,
    rank: int,
    lr: str,
    max_steps: int,
    save_every: int,
    grad_accum: int,
    blocks_to_swap: int,
    cache_batch: int,
    optimizer: str,
    eval_before: bool,
) -> tuple[Path, Path]:
    dataset_config = CONFIG_DIR / f"{run_name}.dataset.toml"
    train_config = CONFIG_DIR / f"{run_name}.toml"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

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
weight_decay = 0.01
eps = 1e-8
"""
    else:
        optimizer_text = f"""[optimizer]
type = 'AdamW8bitKahan'
lr = {lr}
betas = [0.9, 0.99]
weight_decay = 0.01
stabilize = false
"""

    blocks_line = f"blocks_to_swap = {blocks_to_swap}\n" if blocks_to_swap > 0 else ""
    train_text = f"""output_dir = {quote_toml(output_dir)}
dataset = {quote_toml(dataset_config.as_posix())}

epochs = 1000
max_steps = {max_steps}
micro_batch_size_per_gpu = 1
pipeline_stages = 1
gradient_accumulation_steps = {grad_accum}
gradient_clipping = 1.0
warmup_steps = 100
eval_before_first_step = {str(eval_before).lower()}
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

    write_file(dataset_config, dataset_text)
    write_file(train_config, train_text)
    return dataset_config, train_config


def main() -> int:
    print("Krea2 diffusion-pipe wizard")
    print("Dataset format: image/video files with matching .txt captions in the same folder.")
    print("Windows paths like C:\\ai\\input are accepted and converted to /mnt/c/ai/input.")
    print("If a required path is wrong, choose retry or cancel. Cancel exits without writing configs.")

    run_name = ask("\nRun name", "krea2_test")
    dataset_path = choose_dir("Training image folder", "/mnt/c/ai/input")
    output_dir = choose_output_dir(f"{DEFAULT_OUTPUT_DIR}/{run_name}")

    diffusion_model = choose_file(
        "Krea2 base diffusion model, e.g. krea2_raw.safetensors",
        ["krea2", "raw"],
        "/mnt/c/ai/models/diffusion_models/krea2_raw.safetensors",
        exclude_parts=["/loras/", "/output/", "/training_runs/"],
    )
    vae = choose_file(
        "Qwen Image VAE, e.g. qwen_image_vae.safetensors",
        ["qwen", "vae"],
        find_vae("qwen_image_vae.safetensors"),
        exclude_parts=["/loras/", "/output/", "/training_runs/"],
    )
    text_encoder = choose_file(
        "Qwen3VL text encoder, e.g. qwen3vl_4b_bf16.safetensors",
        ["qwen3vl"],
        find_text_encoder("qwen_3vl_4b"),
        exclude_parts=["/loras/", "/output/", "/training_runs/"],
    )

    resolution = ask_int("\nResolution", 512)
    min_ar = ask_float("Min aspect ratio", 0.5)
    max_ar = ask_float("Max aspect ratio", 2.0)
    num_ar_buckets = ask_int("AR bucket count", 7)
    repeats = ask_int("Dataset repeats, num_repeats", 1)
    trigger_word = ask("Trigger word / caption prefix (blank to skip)", "")
    caption_prefix = f"{trigger_word.strip()}, " if trigger_word.strip() else ""

    rank = ask_int("\nLoRA rank", 16)
    lr = ask("Learning rate", "1e-4")
    max_steps = ask_int("Max steps", 1000)
    save_every = ask_int("Save every N steps", 250)
    grad_accum = ask_int("Gradient accumulation steps", 4)
    blocks_to_swap = ask_int("blocks_to_swap, increase if VRAM is tight", 16)
    cache_batch = ask_int("Caching batch size", 1)

    optimizer = ask("Optimizer: adamw8bitkahan or adamw_optimi", "adamw8bitkahan").lower()
    if optimizer not in ("adamw8bitkahan", "adamw_optimi"):
        print("Unknown optimizer. Falling back to adamw8bitkahan.")
        optimizer = "adamw8bitkahan"

    eval_before = ask_bool("Run eval before first step", False)
    cache_only = warn_before_gpu_work("cache_only")
    start_train = warn_before_gpu_work("training")

    dataset_config, train_config = build_configs(
        run_name,
        dataset_path,
        output_dir,
        diffusion_model,
        vae,
        text_encoder,
        resolution,
        min_ar,
        max_ar,
        num_ar_buckets,
        repeats,
        caption_prefix,
        rank,
        lr,
        max_steps,
        save_every,
        grad_accum,
        blocks_to_swap,
        cache_batch,
        optimizer,
        eval_before,
    )

    cmd_base = ["deepspeed", "--num_gpus=1", "train.py", "--deepspeed", "--config", train_config.as_posix()]
    cache_cmd = cmd_base + ["--cache_only"]
    train_cmd = cmd_base

    print("\nGenerated configs")
    print(f"  dataset config: {dataset_config}")
    print(f"  train config:   {train_config}")
    print("\nCommands")
    print("  cache:", " ".join(shlex.quote(x) for x in cache_cmd))
    print("  train:", " ".join(shlex.quote(x) for x in train_cmd))

    if cache_only:
        print("\nStarting cache_only.")
        subprocess.run(cache_cmd, cwd=REPO_DIR, check=False)

    if start_train:
        print("\nStarting training.")
        return subprocess.run(train_cmd, cwd=REPO_DIR).returncode

    print("\nRun later from Windows:")
    print(f"  C:\\ai\\diffusion-pipe\\run_wsl.bat deepspeed --num_gpus=1 train.py --deepspeed --config {train_config.as_posix()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UserCancelled:
        print("\nCancelled. No config was written. Re-run krea2_wizard.bat when ready.")
        raise SystemExit(2)
    except KeyboardInterrupt:
        print("\nCancelled by Ctrl+C. No config was written.")
        raise SystemExit(130)
