#!/usr/bin/env python3
import subprocess
import webbrowser
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LOGDIR = REPO_DIR / "training_runs"


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


def main() -> int:
    print("diffusion-pipe TensorBoard launcher")
    print("Use the default to see all runs, or paste a specific run folder.")
    print("Windows paths like \"C:\\ai\\diffusion-pipe\\training_runs\\krea2\\JE_KREA2\" are accepted.\n")

    logdir = Path(win_to_wsl(ask("Log directory", str(DEFAULT_LOGDIR))))
    if not logdir.exists():
        print(f"Missing log directory: {logdir}")
        return 1

    port = ask("Port", "6006")
    if not port.isdigit():
        print("Port must be a number.")
        return 1

    url = f"http://localhost:{port}"
    print(f"\nStarting TensorBoard: {url}")
    print("Press Ctrl+C in this window to stop TensorBoard.\n")

    if ask_bool("Open browser", True):
        webbrowser.open(url)

    cmd = [
        "tensorboard",
        "--logdir",
        str(logdir),
        "--host",
        "0.0.0.0",
        "--port",
        port,
    ]
    return subprocess.call(cmd, cwd=REPO_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
