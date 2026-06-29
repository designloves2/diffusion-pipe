#!/usr/bin/env python3
import subprocess
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_DIR / "configs" / "generated" / "krea2"


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


def list_configs() -> list[Path]:
    if not CONFIG_DIR.exists():
        return []
    return sorted(p for p in CONFIG_DIR.glob("*.toml") if not p.name.endswith(".dataset.toml"))


def choose_config() -> Path:
    configs = list_configs()
    print("Krea2 config runner")
    print(f"Config folder: {CONFIG_DIR}")
    if configs:
        print("")
        for idx, path in enumerate(configs, start=1):
            print(f"  {idx}. {path.name}")
    else:
        print("No generated Krea2 configs found.")

    while True:
        default = "1" if configs else ""
        value = ask("\nSelect config number or enter TOML path", default)
        if value.isdigit() and configs:
            idx = int(value)
            if 1 <= idx <= len(configs):
                return configs[idx - 1]
            print("Invalid config number.")
            continue
        path = Path(value.replace("\\", "/"))
        if len(value) >= 3 and value[1:3] in (":/", ":\\"):
            drive = value[0].lower()
            rest = value[3:].replace("\\", "/")
            path = Path(f"/mnt/{drive}/{rest}")
        if path.is_file():
            return path
        print(f"Missing config file: {path}")


def choose_action() -> list[str]:
    print("")
    print("Actions")
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
            checkpoint = ask("Checkpoint folder name, e.g. 20260212_07-06-40")
            return ["--resume_from_checkpoint", checkpoint]
        print("Please select 1, 2, 3, or 4.")


def main() -> int:
    config = choose_config()
    extra_args = choose_action()

    print("")
    print("Before running, make sure AI-Toolkit/ComfyUI/other training jobs are not using the GPU.")
    if not ask_bool("Run now", False):
        print("Cancelled. Nothing was started.")
        return 2

    cmd = [
        "deepspeed",
        "--num_gpus=1",
        "train.py",
        "--deepspeed",
        "--config",
        config.as_posix(),
        *extra_args,
    ]
    print("")
    print("Running:")
    print(" ".join(cmd))
    return subprocess.run(cmd, cwd=REPO_DIR).returncode


if __name__ == "__main__":
    raise SystemExit(main())
