# Makefile for HY-Motion-1.0

This Makefile provides a convenient way to build, run, and manage the HY-Motion-1.0 project.

## Usage

### Basic Commands

```bash
# Show help
make help

# Install Python dependencies
make setup

# Install package in development mode
make install

# Run CLI inference
make run-cli

# Run Gradio web interface
make run-gradio
```

### Model Management

```bash
# Download all required models
make download-models

# Download specific model components
make download-t2m          # Text-to-motion models
make download-encoders     # Text encoder models (CLIP, Qwen)
make download-rewriter     # Prompt rewriter model
```

### Docker Support

```bash
# Build Docker image
make docker-build

# Run Docker container with GPU support
make docker-run
```

### Cleanup

```bash
# Clean up temporary files
make clean
```

## Environment Variables

The Makefile uses the following environment variables:

- `USE_HF_MODELS=0`: Forces the use of local model checkpoints instead of downloading from Hugging Face

## Target Details

### `make setup`
Installs all Python dependencies from `requirements.txt`. This includes:
- PyTorch and related libraries
- Hugging Face libraries (transformers, diffusers)
- Motion processing dependencies
- Gradio for web interface

### `make install`
Installs the HY-Motion-1.0 package in development mode using `pip install -e .`.

### `make download-models`
Downloads all required models to the `ckpts/` directory:
- HY-Motion-1.0 (standard model)
- HY-Motion-1.0-Lite (lightweight model)
- CLIP text encoder
- Qwen text encoder
- Prompt rewriter model

### `make run-cli`
Runs the command-line inference tool with the standard HY-Motion-1.0 model.

### `make run-gradio`
Launches the Gradio web interface for interactive motion generation.

### `make docker-build`
Builds a Docker image with all dependencies and CUDA support.

### `make docker-run`
Runs the Docker container with GPU access and port forwarding for the Gradio interface.

## Notes

- The Makefile assumes you have `huggingface-cli` installed for model downloads
- Docker commands require Docker to be installed and configured with GPU support
- All commands set `USE_HF_MODELS=0` to use local checkpoints when available

## Docker Troubleshooting

### Python not found in Docker container

If you encounter issues with Python not being available in the Docker container:

1. **Use the dedicated Python targets**:
   ```bash
   make docker-python     # Direct Python shell
   make docker-python3    # Direct Python3 shell
   ```

2. **Check Python versions**:
   ```bash
   make docker-run        # This will show Python versions before starting bash
   ```

3. **Manual fix inside container**:
   ```bash
   # If you're already in the container and python doesn't work:
   ln -s /usr/bin/python3 /usr/bin/python
   export PYTHONPATH=/app
   ```

### GPU not detected

Make sure you have:
- NVIDIA drivers installed on your host
- NVIDIA Container Toolkit installed
- Docker configured for GPU support

### Volume mounting issues

The Makefile mounts:
- Current directory to `/app`
- `ckpts/` directory to `/app/ckpts`

Make sure these directories exist and have proper permissions.