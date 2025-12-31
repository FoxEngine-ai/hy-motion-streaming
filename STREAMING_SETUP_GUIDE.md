# HY-Motion Streaming Setup Guide

## 🚀 Quick Start with Makefile

### 1. Setup Streaming Environment

```bash
# Install dependencies and setup directories
make setup-streaming

# Download models (if needed)
make download-t2m

# Fix checkpoint paths automatically
make fix-checkpoint-paths

# Or do everything in one command
make streaming-setup
```

### 2. Run the Streaming App

#### Local Execution
```bash
make run-gradio-streaming
```

#### Docker Execution
```bash
# Build the Docker image first
docker build -t hymotion:latest .

# Then run the streaming app
make run-gradio-streaming-docker
```

## 🔧 Manual Setup (if needed)

### Checkpoint Directory Structure

The app looks for checkpoints in these locations (in order):

1. `ckpts/tencent/HY-Motion-1.0/` (primary location)
2. `downloaded_models/HY-Motion-1.0-Lite/` (alternative)
3. `downloaded_models/HY-Motion-1.0/` (alternative)
4. `ckpts/HY-Motion-1.0-Lite/` (alternative)

### Expected Files
```
ckpts/tencent/HY-Motion-1.0/
├── config.yml      # Configuration file
└── latest.ckpt     # Model checkpoint
```

### Create Symlinks (if checkpoints are elsewhere)

```bash
# If you have checkpoints in downloaded_models/HY-Motion-1.0-Lite
ln -s ../downloaded_models/HY-Motion-1.0-Lite ckpts/tencent/HY-Motion-1.0

# If you have checkpoints in downloaded_models/HY-Motion-1.0  
ln -s ../downloaded_models/HY-Motion-1.0 ckpts/tencent/HY-Motion-1.0
```

## 🐳 Docker Specific Setup

### Volume Mounts

```bash
docker run -it --rm \
    --gpus all \
    -p 7860:7860 -p 7861:7861 \
    -v $(pwd)/ckpts:/app/ckpts \
    -v $(pwd)/downloaded_models:/app/downloaded_models \
    -e USE_HF_MODELS=0 \
    -e PYTHONPATH=/app \
    hymotion:latest python gradio_app_streaming.py
```

### Port Configuration

- **Default Port**: 7860
- **Fallback Ports**: 7861, 7862, etc. (automatic selection)
- **Custom Port**: Use `--port 7862` argument

## 🔄 Troubleshooting

### "Model files not found" Error

**Solution 1**: Download models
```bash
make download-t2m
```

**Solution 2**: Create directory structure
```bash
mkdir -p ckpts/tencent/HY-Motion-1.0
# Copy your config.yml and latest.ckpt files there
```

**Solution 3**: Use symlinks
```bash
make fix-checkpoint-paths
```

### Port Already in Use

The app automatically finds an available port. If you want to specify:
```bash
python gradio_app_streaming.py --port 7862
```

## 📋 Available Makefile Targets

```bash
# Show all available targets
make help

# Streaming-specific targets
make setup-streaming          # Setup streaming environment
make fix-checkpoint-paths     # Fix checkpoint paths
make streaming-setup          # Complete streaming setup
make run-gradio-streaming     # Run streaming app locally
make run-gradio-streaming-docker # Run streaming app in Docker
```

## ✨ Features

- **Automatic Checkpoint Detection**: Looks in multiple locations
- **Automatic Port Selection**: Handles port conflicts
- **Docker Support**: Volume mounts and environment variables
- **Makefile Integration**: Easy setup and execution
- **Gradio 5.38.2**: Updated and compatible

## 🎬 Running the Streaming Interface

Once everything is set up, access the interface at:
- **Local**: `http://localhost:7860`
- **Docker**: `http://0.0.0.0:7860`

The streaming interface provides real-time frame-by-frame motion generation with progress visualization!