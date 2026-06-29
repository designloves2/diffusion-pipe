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
            force_stop = gr.Button("Force stop", variant="stop")
            attach_log = gr.Button("Attach latest UI log")

        status = gr.Textbox(label="Status", lines=4)
        log = gr.Textbox(label="Training log tail", lines=18)

        with gr.Row():
            gpu = gr.Button("GPU/process status")
            runs = gr.Button("Recent run files")
            checkpoints = gr.Button("Saved checkpoints")

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
        force_stop.click(force_stop_training, outputs=[status, log])
        attach_log.click(attach_latest_log, outputs=[status, log])
        gpu.click(gpu_status, outputs=info)
        runs.click(recent_runs, outputs=info)
        checkpoints.click(saved_checkpoints, inputs=config, outputs=info)
        tb_start.click(start_tensorboard, inputs=[tb_logdir, tb_port], outputs=[tb_status, tb_embed])
        tb_stop.click(stop_tensorboard, outputs=[tb_status, tb_embed])

    return app


def main() -> None:
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, inbrowser=False)


if __name__ == "__main__":
    main()
