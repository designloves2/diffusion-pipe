#!/usr/bin/env python3
import subprocess
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_DIR / "configs" / "generated"


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default not in (None, "") else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return default if value == "" and default is not None else value


def ask_bool(prompt: str, default: bool) -> bool:
    default_text = "y" if default else "n"
    while True:
        value = ask(prompt + " (y/n)", default_text).lower()
        if value in ("y", "yes", "1", "true"):
            return True
        if value in ("n", "no", "0", "false"):
            return False
        print("Please enter y or n.")


def win_to_wsl(value: str) -> str:
    value = value.strip()
    while len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1].strip()
    value = value.strip('"').strip("'").strip()
    if len(value) >= 3 and value[1:3] in (":/", ":\\"):
        rest = value[3:].replace("\\", "/")
        return f"/mnt/{value[0].lower()}/{rest}"
    return value.replace("\\", "/")


def list_configs() -> list[Path]:
    if not CONFIG_ROOT.exists():
        return []
    return sorted(p for p in CONFIG_ROOT.rglob("*.toml") if not p.name.endswith(".dataset.toml"))


def choose_config() -> Path:
    configs = list_configs()
    print("diffusion-pipe generated config runner")
    for idx, path in enumerate(configs, start=1):
        print(f"  {idx}. {path.relative_to(CONFIG_ROOT)}")
    while True:
        value = ask("\nSelect config number or enter TOML path", "1" if configs else "")
        if value.isdigit() and configs:
            idx = int(value)
            if 1 <= idx <= len(configs):
                return configs[idx - 1]
        path = Path(win_to_wsl(value))
        if path.is_file():
            return path
        print(f"Missing config file: {path}")


def choose_action() -> list[str]:
    print("\nActions")
    print("  1. cache_only")
    print("  2. train")
    print("  3. resume latest checkpoint")
    print("  4. resume specific checkpoint folder")
    while True:
        value = ask("Select action", "2")
        if value == "1":
            return ["--cache_only"]
        if value == "2":
            return []
        if value == "3":
            return ["--resume_from_checkpoint"]
        if value == "4":
            return ["--resume_from_checkpoint", ask("Checkpoint folder name")]
        print("Please select 1, 2, 3, or 4.")


def main() -> int:
    config = choose_config()
    extra = choose_action()
    print("\nBefore running, make sure AI-Toolkit/ComfyUI/other training jobs are not using the GPU.")
    if not ask_bool("Run now", False):
        print("Cancelled. Nothing was started.")
        return 2
    cmd = ["deepspeed", "--num_gpus=1", "train.py", "--deepspeed", "--config", config.as_posix(), *extra]
    print("\nRunning:")
    print(" ".join(cmd))
    return subprocess.run(cmd, cwd=REPO_DIR).returncode


if __name__ == "__main__":
    raise SystemExit(main())
