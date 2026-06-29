#!/usr/bin/env python3
import os
import signal
import subprocess
import time
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_DIR / "configs" / "generated"
RUN_ROOT = REPO_DIR / "training_runs"
LOG_ROOT = REPO_DIR / "ui_logs"

TRAIN_PROC: subprocess.Popen | None = None
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


def tail(path: Path | None, lines: int = 80) -> str:
    if path is None or not path.exists():
        return "No log yet."
    data = path.read_text(errors="replace").splitlines()
    return "\n".join(data[-lines:])


def process_alive(proc: subprocess.Popen | None) -> bool:
    return proc is not None and proc.poll() is None


def stop_process(proc: subprocess.Popen | None) -> str:
    if proc is None:
        return "No process was started from this UI."
    if proc.poll() is not None:
        return f"Process already exited with code {proc.returncode}."
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        return "Stop signal sent."
    except ProcessLookupError:
        return "Process already stopped."


def start_training(choice: str, action: str, checkpoint: str) -> tuple[str, str]:
    global TRAIN_PROC, TRAIN_LOG
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
    log_file = TRAIN_LOG.open("w", encoding="utf-8", errors="replace")
    log_file.write("Running:\n" + " ".join(cmd) + "\n\n")
    log_file.flush()
    TRAIN_PROC = subprocess.Popen(
        cmd,
        cwd=REPO_DIR,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )
    return f"Started PID {TRAIN_PROC.pid}. Log: {TRAIN_LOG}", tail(TRAIN_LOG)


def stop_training() -> tuple[str, str]:
    return stop_process(TRAIN_PROC), tail(TRAIN_LOG)


def tensorboard_frame(port: str) -> str:
    port = port if (port or "").isdigit() else "6006"
    return (
        f'<div style="height:760px;border:1px solid #ddd;border-radius:8px;overflow:hidden;">'
        f'<iframe src="http://localhost:{port}" '
        f'style="width:100%;height:100%;border:0;background:white;"></iframe>'
        f'</div>'
    )


def start_tensorboard(logdir: str, port: str) -> tuple[str, str]:
    global TB_PROC, TB_LOG
    if process_alive(TB_PROC):
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
    return stop_process(TB_PROC), ""


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


def refresh_configs():
    return list_configs()


def build_ui():
    gr = require_gradio()
    with gr.Blocks(title="diffusion-pipe local UI") as app:
        gr.Markdown("# diffusion-pipe local UI")
        gr.Markdown("Thin local controls for generated configs, training, cache jobs, GPU status, and TensorBoard.")

        with gr.Row():
            config = gr.Dropdown(label="Config", choices=list_configs(), interactive=True)
            refresh = gr.Button("Refresh configs")

        with gr.Row():
            action = gr.Radio(
                label="Action",
                choices=["train", "cache_only", "resume latest checkpoint", "resume specific checkpoint"],
                value="train",
            )
            checkpoint = gr.Textbox(label="Checkpoint folder", placeholder="Only needed for resume specific checkpoint")

        with gr.Row():
            start = gr.Button("Start", variant="primary")
            stop = gr.Button("Stop job", variant="stop")

        status = gr.Textbox(label="Status", lines=4)
        log = gr.Textbox(label="Training log tail", lines=18)

        with gr.Row():
            gpu = gr.Button("GPU/process status")
            runs = gr.Button("Recent run files")

        info = gr.Textbox(label="Info", lines=12)

        with gr.Row():
            tb_logdir = gr.Textbox(label="TensorBoard logdir", value=str(RUN_ROOT))
            tb_port = gr.Textbox(label="Port", value="6006")

        with gr.Row():
            tb_start = gr.Button("Start TensorBoard")
            tb_stop = gr.Button("Stop TensorBoard")

        tb_status = gr.Textbox(label="TensorBoard status", lines=2)
        tb_embed = gr.HTML(label="TensorBoard")

        refresh.click(refresh_configs, outputs=config)
        start.click(start_training, inputs=[config, action, checkpoint], outputs=[status, log])
        stop.click(stop_training, outputs=[status, log])
        gpu.click(gpu_status, outputs=info)
        runs.click(recent_runs, outputs=info)
        tb_start.click(start_tensorboard, inputs=[tb_logdir, tb_port], outputs=[tb_status, tb_embed])
        tb_stop.click(stop_tensorboard, outputs=[tb_status, tb_embed])

    return app


def main() -> None:
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, inbrowser=False)


if __name__ == "__main__":
    main()
