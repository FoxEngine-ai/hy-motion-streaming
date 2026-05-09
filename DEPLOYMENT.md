# Deployment, quantization, and dev tooling

> **Scope.** Docker setup, the Makefile target reference, and the
> quantization paths (GGUF for the text encoder + 4-bit for the
> prompter). For library install / pip usage see [README.md](README.md);
> for streaming see [STREAMING.md](STREAMING.md).

---

## Docker

Three Dockerfiles ship in this fork, each tuned for a different
priority:

| File | When to use |
|---|---|
| `Dockerfile` | Default — full set of features, GPU support, all dependencies |
| `Dockerfile.fast` | Build-cache-friendly variant for fast iterative rebuilds |
| `Dockerfile.minimal` | Smallest image; trims optional deps, suitable for headless inference servers |

### Prerequisites

- Docker Engine 20.10+
- NVIDIA Container Toolkit (`nvidia-docker2`) for GPU support
- ~30 GB disk for the default image; ~12 GB for `Dockerfile.minimal`

### Build

```bash
# Default
docker build -t hymotion:latest .

# Fast / minimal variants
docker build -f Dockerfile.fast    -t hymotion:fast    .
docker build -f Dockerfile.minimal -t hymotion:minimal .
```

### Run (gradio)

```bash
docker run -it --rm \
    --gpus all \
    -p 7860:7860 -p 7861:7861 \
    -v "$(pwd)/ckpts:/app/ckpts" \
    -v "$(pwd)/downloaded_models:/app/downloaded_models" \
    -e USE_HF_MODELS=0 \
    hymotion:latest python gradio_app_streaming.py
```

Then open `http://localhost:7860`. Port 7861 is the streaming variant's
fallback if 7860 is taken.

### Volume mounts

The container reads model weights from `/app/ckpts` and
`/app/downloaded_models`. Mount your local copies into those paths so
the container doesn't need to redownload on each run.

### Environment variables

The container respects the same env vars as the Python process:

| Var | Purpose |
|---|---|
| `USE_HF_MODELS` | `1` = pull encoders from HuggingFace hub; `0` = use local `ckpts/` paths |
| `HYMOTION_QWEN_PATH` | Override Qwen text-encoder path (introduced by this fork) |
| `HYMOTION_CLIP_PATH` | Override CLIP text-encoder path (introduced by this fork) |
| `HYMOTION_WOODEN_PATH` | Override path to the `dump_wooden` mesh assets |
| `PYTHONPATH=/app` | For the legacy run-from-cwd code path; harmless if also pip-installed |

### Common Docker problems

**"could not select device driver with capabilities: [[gpu]]"**
`nvidia-docker2` not installed or daemon not configured. See the
NVIDIA Container Toolkit install docs.

**Out-of-memory during build**
Use `Dockerfile.fast` or `Dockerfile.minimal`. The default image pulls
all torch + transformers + ssae deps into a single layer.

**Volume permissions on Linux**
Docker runs as root inside the container; if you bind-mount a host dir
and write to it, the resulting files are owned by root on the host.
Either run the container with `--user "$(id -u):$(id -g)"`, or use a
named docker volume.

**`make docker-build` says python not found**
Older base images. Use the bundled `Dockerfile` instead of any local
custom one — it pins a Python version known to work.

For deeper diagnostics, the previous standalone troubleshooting doc
material lives in git history if needed (see commit `d28898c` or
later).

---

## Makefile

The `Makefile` wraps the most common workflows so you don't have to
remember the long form. Run `make help` for the live list of targets.

### Setup

```bash
make setup                    # install deps + create directories
make install                  # pip install -e .
make download-models          # download HY-Motion + Text2MotionPrompter weights
make download-t2m             # download just HY-Motion (skip prompter)
```

### Running

```bash
make run-cli                  # local_infer.py with sensible defaults
make run-gradio               # standard gradio app, port 7860
make run-gradio-streaming     # streaming gradio variant
make run-streaming-server     # headless REST API (streaming_server.py)
```

### Docker shortcuts

```bash
make docker-build             # build the default image
make docker-run               # run with GPU + standard volume mounts
make run-gradio-docker        # gradio inside the container
make run-gradio-streaming-docker  # streaming gradio inside the container
```

### Quantization

```bash
make quantize-prompter        # one-time: build the 4-bit prompter checkpoint
```

See the [Quantization](#quantization) section below.

### Other

```bash
make fix-checkpoint-paths     # rewrite ckpts/ symlinks if your downloads landed elsewhere
make clean                    # remove temporary files, build artefacts
make help                     # full target list
```

### Environment variables

The Makefile respects:

| Var | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `ckpts/tencent/HY-Motion-1.0` | path passed to `local_infer.py` |
| `PORT` | `7860` | gradio port |
| `USE_HF_MODELS` | `0` | toggle HF-hub vs local paths (see Docker section) |

---

## Quantization

Two distinct quantization paths exist in this fork; they apply to
different models and serve different goals.

### Prompter model: 4-bit pre-quantization

**The problem.** The prompter (`Text2MotionPrompter`) is loaded with
on-the-fly bitsandbytes 4-bit quantization on every gradio start, which
costs ~10 minutes per cold start.

**The fix.** Quantize once, save to disk, reuse on every startup. After
this, cold starts drop to ~10 seconds.

```bash
# One-time setup
make quantize-prompter

# Or directly:
python3 scripts/quantize_prompter_model.py
```

This produces `ckpts/Text2MotionPrompter_4bit/`. The runtime
auto-detects the `_4bit` suffix and uses it preferentially over the
unquantized model. No code changes needed in callers.

The script uses:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
```

NF4 + double-quant gives a good tradeoff of size (~75% reduction) vs
quality.

### Text encoder: GGUF + llama-cpp-python

**The problem.** The Qwen3-8B text encoder weighs ~16 GB as fp16
safetensors and pins a chunk of GPU memory regardless of whether the
text encoder is the bottleneck.

**The fix.** Run Qwen3-8B as a GGUF (typically Q4-K or IQ4-XS, ~4-5 GB)
on CPU via `llama-cpp-python`. You give up a few hundred milliseconds
per generation for the text encoder pass; you gain ~12 GB of GPU memory
to use elsewhere (the main motion encoder, for instance).

**Pick a GGUF.** Any Qwen3-8B GGUF works; we test against
[`unsloth/Qwen3-8B-GGUF`](https://huggingface.co/unsloth/Qwen3-8B-GGUF).
Pull just one quantization variant, e.g.:

```bash
huggingface-cli download unsloth/Qwen3-8B-GGUF \
    Qwen3-8B-IQ4_XS.gguf \
    --local-dir ./ckpts/Qwen3-8B-GGUF/
```

**Pull the tokenizer assets.** GGUF files don't carry the
HuggingFace-style tokenizer metadata. Either download them alongside:

```bash
huggingface-cli download Qwen/Qwen3-8B \
    config.json tokenizer.json tokenizer_config.json generation_config.json \
    --local-dir ./ckpts/Qwen3-8B-GGUF/
```

…or let your runtime fetch them on first use via `huggingface_hub`. The
Voxta integration does the latter.

**Wire it up.** Pass `use_gguf=True` to `T2MRuntime`, and point
`HYMOTION_QWEN_PATH` at the directory with the `.gguf` file:

```bash
export HYMOTION_QWEN_PATH=./ckpts/Qwen3-8B-GGUF
```

```python
runtime = T2MRuntime(
    config_path="...config.yml",
    ckpt_name="latest.ckpt",
    use_gguf=True,
)
```

The text-encoder path will discover the `.gguf` file via
`os.walk(HYMOTION_QWEN_PATH)`, instantiate
`llama_cpp.Llama(... embeddings=True)`, and route Qwen embedding
requests through it instead of through transformers.

### Combined: 4-bit prompter + GGUF text encoder

Both paths compose. Common configurations:

| Goal | Prompter | Qwen | GPU footprint |
|---|---|---|---|
| Cold-start latency | full fp16 | full fp16 | maximum |
| Production headless | 4-bit | full fp16 | moderate |
| GPU-poor / CPU offload | 4-bit | GGUF on CPU | minimum |
| Voxta default | 4-bit (lazy) | GGUF on CPU | minimum |

---

## Related docs

- [README.md](README.md) — install, library usage, model weights
- [STREAMING.md](STREAMING.md) — streaming gradio + motion continuation + REST API
