# Docker Setup for HY-Motion 1.0

This guide provides instructions for building and running HY-Motion 1.0 in a Docker container with Python 3.12 and CUDA support.

## Prerequisites

- Docker installed on your system
- NVIDIA Container Toolkit installed (for GPU support)
- At least 8GB of GPU memory (recommended: 16GB+ for better performance)

## Building the Docker Image

1. **Install NVIDIA Container Toolkit** (if not already installed):
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

2. **Build the Docker image**:
   ```bash
   docker build -t hymotion:latest .
   ```

   This will create a Docker image with:
   - Ubuntu 22.04 base
   - CUDA 12.1.1
   - Python 3.12.0
   - All required dependencies for HY-Motion 1.0

## Running the Container

### Basic Usage

```bash
# Run with GPU support
docker run --gpus all -it --rm \
  -v $(pwd):/app \
  -v /path/to/ckpts:/app/ckpts \
  -v /path/to/output:/app/output \
  --name hymotion \
  hymotion:latest
```

### Running Specific Commands

**Local inference**:
```bash
docker run --gpus all -it --rm \
  -v $(pwd):/app \
  -v /path/to/ckpts:/app/ckpts \
  -v /path/to/output:/app/output \
  hymotion:latest \
  python3 local_infer.py --model_path ckpts/tencent/HY-Motion-1.0
```

**Gradio app**:
```bash
docker run --gpus all -it --rm \
  -p 7860:7860 \
  -v $(pwd):/app \
  -v /path/to/ckpts:/app/ckpts \
  hymotion:latest \
  python3 gradio_app.py
```

## Volume Mounts

- `$(pwd):/app` - Mounts the current directory to /app in the container
- `/path/to/ckpts:/app/ckpts` - Mounts your checkpoints directory
- `/path/to/output:/app/output` - Mounts your output directory for generated motions

## Environment Variables

You can set environment variables when running the container:
```bash
docker run --gpus all -it --rm \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTHONPATH=/app \
  hymotion:latest
```

## Troubleshooting

### CUDA Errors
If you encounter CUDA errors:
1. Ensure you have the latest NVIDIA drivers installed
2. Verify the NVIDIA Container Toolkit is properly installed
3. Check that your GPU supports CUDA 12.1

### Build Issues
If the build fails:
1. Check your internet connection (some packages are large)
2. Ensure you have enough disk space (the image is several GB)
3. Try building with `--no-cache` flag

### Performance Issues
For better performance:
- Use a GPU with at least 16GB memory
- Ensure your host system has enough RAM
- Consider using `--shm-size=8g` for larger shared memory

## Updating the Container

To update to the latest version:
```bash
docker pull hymotion:latest
docker rmi hymotion:old  # Remove old images if needed
```

## Notes

- The container includes Python 3.12.0 compiled from source
- CUDA 12.1.1 is included with cuDNN and other libraries
- PyTorch 2.5.1 with CUDA 12.1 support is installed
- All dependencies from requirements.txt are included