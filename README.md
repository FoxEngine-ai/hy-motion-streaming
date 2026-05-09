[中文阅读](README_zh_cn.md)


<p align="center">
  <img src="./assets/banner.png" alt="Banner" width="100%">
</p>

<div align="center">
  <a href="https://hunyuan.tencent.com/motion" target="_blank">
    <img src="https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage" height="22px" alt="Official Site">
  </a>
  <a href="https://github.com/Tencent-Hunyuan/HY-Motion-1.0" target="_blank">
    <img src="https://img.shields.io/badge/Upstream-Tencent--Hunyuan-181717?logo=github&logoColor=white" height="22px" alt="Upstream Repo">
  </a>
  <a href="https://github.com/FoxEngine-ai/hy-motion-streaming" target="_blank">
    <img src="https://img.shields.io/badge/Fork-FoxEngine--ai-1F6FEB?logo=github&logoColor=white" height="22px" alt="Fork">
  </a>
  <a href="https://huggingface.co/spaces/tencent/HY-Motion-1.0" target="_blank">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Demo-276cb4.svg" height="22px" alt="HuggingFace Space">
  </a>
  <a href="https://huggingface.co/tencent/HY-Motion-1.0" target="_blank">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg" height="22px" alt="HuggingFace Models">
  </a>
  <a href="https://arxiv.org/pdf/2512.23464" target="_blank">
    <img src="https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv" height="22px" alt="ArXiv Report">
  </a>
</div>


# HY-Motion 1.0 — FoxEngine streaming fork

> **Fork notice.** This repository is a downstream fork of
> [Tencent-Hunyuan/HY-Motion-1.0](https://github.com/Tencent-Hunyuan/HY-Motion-1.0)
> maintained by [FoxEngine.ai](https://github.com/FoxEngine-ai). All credit
> for the underlying model, training pipeline, and research belongs to the
> original Tencent Hunyuan 3D Digital Human Team.
>
> This fork adds the production-shape glue we needed for live integrations
> (Voxta in particular): pip-installability, real-time frame streaming,
> motion continuation across clips, GGUF / 4-bit quantization paths, and
> Docker deployment. The core inference code is unchanged.

## What this fork adds

| Feature | What it does | Where to read more |
|---|---|---|
| **Pip-installable** | `pip install git+...` produces a usable library — assets bundled, paths resolved relative to package | this README |
| **Real-time streaming** | Generates motion frame-by-frame and streams to the UI as it goes, instead of holding everything until generation completes | [STREAMING.md](STREAMING.md) |
| **Motion continuation** | Use the last frames of one generation as the seed for the next — chain "walk → run → jump" without seam discontinuities | [STREAMING.md](STREAMING.md#motion-continuation) |
| **GGUF text encoder** | Run Qwen3-8B as a 4 GB GGUF on CPU via llama-cpp-python instead of the full 16 GB safetensors download | [DEPLOYMENT.md](DEPLOYMENT.md#quantization) |
| **4-bit / 8-bit quantization** | bitsandbytes-backed quantization for the prompter model + main motion encoder | [DEPLOYMENT.md](DEPLOYMENT.md#quantization) |
| **REST API server** | Headless inference endpoint (`streaming_server.py`) for embedding into other applications | [STREAMING.md](STREAMING.md#rest-api) |
| **Docker + Makefile** | Reproducible builds for deployment, with `make` shortcuts for the common workflows | [DEPLOYMENT.md](DEPLOYMENT.md) |

## Install

```bash
pip install git+https://github.com/FoxEngine-ai/hy-motion-streaming.git
```

This installs the `hymotion` Python package along with its dependencies. The
wooden-mesh assets needed by `WoodenMesh` ship inside the wheel so no
auxiliary asset download is required for that path.

You still need to download the **HY-Motion model checkpoint** and the
**Qwen3-8B text encoder** separately — they're large and not bundled with
the package. See [Model weights](#model-weights) below.

For development:

```bash
git clone https://github.com/FoxEngine-ai/hy-motion-streaming.git
cd hy-motion-streaming
pip install -e .
```

## Library usage (quick start)

```python
from hymotion.utils.t2m_runtime import T2MRuntime

runtime = T2MRuntime(
    config_path="path/to/HY-Motion-1.0/config.yml",
    ckpt_name="latest.ckpt",                    # bare filename, resolved against config_path's dir
    device_ids=[0],                             # GPU ids; pass an empty list or use force_cpu=True for CPU
    use_gguf=True,                              # GGUF path for the Qwen text encoder (CPU-friendly)
    disable_prompt_engineering=True,            # skip the OpenAI-backed prompt rewriter
)

html, files, motion_dict = runtime.generate_motion(
    prompt="a person waving",
    duration_seconds=3.0,
    cfg_scale=5.0,
    seed=42,
)

# motion_dict carries rot6d, transl, keypoints3d for the generated frames
```

To override the text-encoder paths without editing code, set environment
variables before starting the process:

```bash
export HYMOTION_QWEN_PATH=/path/to/your/Qwen3-8B-GGUF/
export HYMOTION_CLIP_PATH=openai/clip-vit-large-patch14   # or a local path
```

## Model weights

The HY-Motion checkpoint and Qwen3 / CLIP text encoders are downloaded
separately:

| Asset | Source | Notes |
|---|---|---|
| HY-Motion-1.0 (1B) | [HF: tencent/HY-Motion-1.0](https://huggingface.co/tencent/HY-Motion-1.0/tree/main/HY-Motion-1.0) | Standard model, ~45 GB |
| HY-Motion-1.0-Lite (0.46B) | [HF: tencent/HY-Motion-1.0/HY-Motion-1.0-Lite](https://huggingface.co/tencent/HY-Motion-1.0/tree/main/HY-Motion-1.0-Lite) | Lightweight variant |
| Qwen3-8B (full) | [HF: Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 16 GB safetensors |
| Qwen3-8B (GGUF, 4-bit) | [HF: unsloth/Qwen3-8B-GGUF](https://huggingface.co/unsloth/Qwen3-8B-GGUF) | ~5 GB, runs on CPU via llama-cpp-python |
| CLIP ViT-L/14 | [HF: openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) | Auto-downloaded by transformers if path is the hub ID |

See [`ckpts/README.md`](ckpts/README.md) for the original Tencent download
instructions. Tools like `huggingface-cli download` work fine.

---

# Upstream documentation

The remainder of this file is the upstream Tencent README, preserved
verbatim — the model architecture, training pipeline, and research credits
remain unchanged from Tencent's release.

## **Introduction**

**HY-Motion 1.0** is a series of text-to-3D human motion generation models based on Diffusion Transformer (DiT) and Flow Matching. It allows developers to generate skeleton-based 3D character animations from simple text prompts, which can be directly integrated into various 3D animation pipelines. This model series is the first to scale DiT-based text-to-motion models to the billion-parameter level, achieving significant improvements in instruction-following capabilities and motion quality over existing open-source models.

### Key Features
- **State-of-the-Art Performance**: Achieves state-of-the-art performance in both instruction-following capability and generated motion quality.

- **Billion-Scale Models**: We are the first to successfully scale DiT-based models to the billion-parameter level for text-to-motion generation. This results in superior instruction understanding and following capabilities, outperforming comparable open-source models.

- **Advanced Three-Stage Training**: Our models are trained using a comprehensive three-stage process:

    - *Large-Scale Pre-training*: Trained on over 3,000 hours of diverse motion data to learn a broad motion prior.

    - *High-Quality Fine-tuning*: Fine-tuned on 400 hours of curated, high-quality 3D motion data to enhance motion detail and smoothness.

    - *Reinforcement Learning*: Utilizes Reinforcement Learning from human feedback and reward models to further refine instruction-following and motion naturalness.



<p align="center">
  <img src="./assets/pipeline.png" alt="System Overview" width="100%">
</p>

<p align="center">
  <img src="./assets/arch.png" alt="Architecture" width="100%">
</p>

<p align="center">
  <img src="./assets/sotacomp.jpg" alt="ComparisonSoTA" width="100%">
</p>


## 🎁 Model Zoo

**HY-Motion 1.0 Series**

| Model | Description | Date | Size | Huggingface |
|:-------|:-------------|:------:|:------:|:-------------:|
| **HY-Motion-1.0** | Standard Text to Motion Generation Model | 2025-12-30 | 1.0B | [Download](https://huggingface.co/tencent/HY-Motion-1.0/tree/main/HY-Motion-1.0) |
| **HY-Motion-1.0-Lite** | Lightweight Text to Motion Generation Model | 2025-12-30 | 0.46B | [Download](https://huggingface.co/tencent/HY-Motion-1.0/tree/main/HY-Motion-1.0-Lite) |


## 🤗 Get Started with HY-Motion 1.0 (upstream gradio demo)

> The instructions below run Tencent's original gradio demo from a source
> checkout. For library / pip-install usage, see [Library usage](#library-usage-quick-start)
> further up. For streaming, REST API, or Docker deployment, see the
> dedicated guides linked at the top of this README.

HY-Motion 1.0 supports macOS, Windows, and Linux.

### Installation

First, install PyTorch via the [official site](https://pytorch.org/). Then install the dependencies:

```bash
pip install -r requirements.txt
```

### Download Model Weights

Please follow the instructions in [ckpts/README.md](ckpts/README.md) to download the necessary model weights.

### Code Usage (CLI)

We provide a script for local batch inference, suitable for processing large amounts of prompts.

```bash
# HY-Motion-1.0
python3 local_infer.py --model_path ckpts/tencent/HY-Motion-1.0

# HY-Motion-1.0-Lite
python3 local_infer.py --model_path ckpts/tencent/HY-Motion-1.0-Lite
```

**Common Parameters:**
- `--input_text_dir`: Directory containing `.txt` or `.json` prompt files.
- `--output_dir`: Directory to save results (default: `output/local_infer`).
- `--disable_duration_est`: Disable LLM-based duration estimation.
- `--disable_rewrite`: Disable LLM-based prompt rewriting.
- `--prompt_engineering_host` / `--prompt_engineering_model_path`: (Optional) Host address / local checkpoint for the Duration Prediction & Prompt Rewrite Module.
    - **Download**: You can download the Duration Prediction & Prompt Rewrite Module from [Here](https://huggingface.co/Text2MotionPrompter/Text2MotionPrompter).
    - **Note**: If you **do not** set these  parameter, you must also set `--disable_duration_est` and `--disable_rewrite`. Otherwise, the script will raise an error due to host unavailable.

### Gradio App

You can host a [Gradio](https://www.gradio.app/) web interface on your local machine for interactive visualization:

```bash
python3 gradio_app.py
```

After running the command, open your browser and visit `http://localhost:7860`.

For the **streaming variant** of the gradio app added by this fork, see [STREAMING.md](STREAMING.md).


## 🔗 BibTeX

If you found this repository helpful, please cite the upstream report:

```bibtex
@article{hymotion2025,
  title={HY-Motion 1.0: Scaling Flow Matching Models for Text-To-Motion Generation},
  author={Tencent Hunyuan 3D Digital Human Team},
  journal={arXiv preprint arXiv:2512.23464},
  year={2025}
}
```

## Acknowledgements

We would like to thank the contributors to the [FLUX](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers), [HuggingFace](https://huggingface.co), [SMPL](https://smpl.is.tue.mpg.de/)/[SMPLH](https://mano.is.tue.mpg.de/), [CLIP](https://github.com/openai/CLIP), [Qwen3](https://github.com/QwenLM/Qwen3), [PyTorch3D](https://github.com/facebookresearch/pytorch3d), [kornia](https://github.com/kornia/kornia), [transforms3d](https://github.com/matthew-brett/transforms3d), [FBX-SDK](https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-2020-0), [GVHMR](https://zju3dv.github.io/gvhmr/), and [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo) repositories or tools, for their open research and exploration.
