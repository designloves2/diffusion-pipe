# Windows Easy Install

This repo includes a one-click Windows + WSL installer for diffusion-pipe.

## Before Running

Install these first:

- NVIDIA driver
- WSL2 with Ubuntu or another Linux distro
- Git for Windows, if you cloned this repo with git

If WSL is not installed, open PowerShell as Administrator and run:

```powershell
wsl --install
```

## Install

From the repo folder, double-click:

```text
easy_install.bat
```

The installer creates:

```text
C:\ai\micromamba
C:\ai\micromamba\envs\diffusion-pipe
```

It installs Python, PyTorch CUDA, DeepSpeed, diffusion-pipe requirements, Gradio, and TensorBoard.

## Launch

After installation, use:

```text
dp_ui.bat
```

Useful helpers:

```text
dp_wizard.bat          Multi-model config wizard
dp_run.bat             Run generated configs
dp_tensorboard.bat     TensorBoard launcher
AITK_to_diffusion_pipe.bat
```

Windows paths such as `"C:\AI\datasets\my_set"` and `"M:\Models\..."` are accepted by the helper scripts.

## Notes

- Native Windows DeepSpeed is not the target; this installer runs diffusion-pipe through WSL.
- Existing model files are not downloaded automatically. Put models, VAEs, and text encoders in your preferred folders and select them in the wizard.
- If a job is already using the GPU, stop it before starting cache or training.
