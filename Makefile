# Makefile for HY-Motion-1.0
# 
# Usage:
#   make help                  - Show this help
#   make setup                 - Install Python dependencies
#   make install               - Install package in development mode
#   make download-models       - Download all required models
#   make download-t2m          - Download text-to-motion models
#   make download-encoders     - Download text encoder models
#   make download-rewriter     - Download prompt rewriter model
#   make run-cli               - Run CLI inference
#   make run-gradio           - Run Gradio web interface
#   make clean                 - Clean up temporary files
#   make docker-build          - Build Docker image
#   make docker-run            - Run Docker container

.PHONY: help setup install download-models download-t2m download-encoders download-rewriter run-cli run-gradio clean docker-build docker-run

# Default target
all: help

# Show help
help:
	@echo "HY-Motion-1.0 Makefile Usage:"
	@echo ""
	@echo "Setup and Installation:"
	@echo "  make setup                 - Install Python dependencies"
	@echo "  make install               - Install package in development mode"
	@echo "  make setup-streaming       - Setup streaming environment"
	@echo "  make fix-checkpoint-paths  - Fix checkpoint paths for streaming"
	@echo "  make streaming-setup       - Complete streaming setup (download + fix paths)"
	@echo ""
	@echo "Model Downloads:"
	@echo "  make download-models       - Download all required models"
	@echo "  make download-t2m          - Download text-to-motion models"
	@echo "  make download-encoders     - Download text encoder models"
	@echo "  make download-rewriter     - Download prompt rewriter model"
	@echo ""
	@echo "Running the Application:"
	@echo "  make run-cli               - Run CLI inference (default: HY-Motion-1.0)"
	@echo "  make run-gradio           - Run Gradio web interface"
	@echo "  make run-gradio-streaming - Run Streaming Gradio web interface"
	@echo "  make run-gradio-streaming-docker - Run Streaming Gradio with Docker"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build          - Build Docker image"
	@echo "  make docker-run            - Run Docker container with bash"
	@echo "  make docker-python         - Run Docker container with Python shell"
	@echo "  make docker-python3        - Run Docker container with Python3 shell"
	@echo ""
	@echo "Testing:"
	@echo "  make test-docker-deps      - Test Docker dependencies locally"
	@echo "  make test-docker-deps-in-container - Test dependencies in Docker container"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean                 - Clean up temporary files"

# Install Python dependencies
setup:
	@echo "Installing Python dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt

# Install package in development mode
install:
	@echo "Installing HY-Motion-1.0 in development mode..."
	pip install -e .

# Download all models
download-models: download-t2m download-encoders download-rewriter

# Download text-to-motion models
download-t2m:
	@echo "Downloading text-to-motion models..."
	mkdir -p ckpts/tencent
	huggingface-cli download tencent/HY-Motion-1.0 --local-dir ckpts/tencent/HY-Motion-1.0 || echo "Failed to download HY-Motion-1.0"

# Download text encoder models
download-encoders:
	@echo "Downloading text encoder models..."
	mkdir -p ckpts
	huggingface-cli download openai/clip-vit-large-patch14 --local-dir ckpts/clip-vit-large-patch14 || echo "Failed to download CLIP encoder"
	huggingface-cli download Qwen/Qwen3-8B --local-dir ckpts/Qwen3-8B || echo "Failed to download Qwen encoder"

# Download prompt rewriter model
download-rewriter:
	@echo "Downloading prompt rewriter model..."
	mkdir -p ckpts
	huggingface-cli download Text2MotionPrompter/Text2MotionPrompter --local-dir ckpts/Text2MotionPrompter || echo "Failed to download prompt rewriter"

# Run CLI inference
run-cli:
	@echo "Running CLI inference with HY-Motion-1.0..."
	USE_HF_MODELS=0 python3 local_infer.py --model_path ckpts/tencent/HY-Motion-1.0

# Run CLI inference with Lite model
run-cli-lite:
	@echo "Running CLI inference with HY-Motion-1.0-Lite..."
	USE_HF_MODELS=0 python3 local_infer.py --model_path ckpts/tencent/HY-Motion-1.0-Lite

# Run Gradio web interface
run-gradio:
	@echo "Running Gradio web interface..."
	USE_HF_MODELS=0 python3 gradio_app.py

# Run Streaming Gradio web interface
run-gradio-streaming:
	@echo "Running Streaming Gradio web interface..."
	USE_HF_MODELS=0 python3 gradio_app_streaming.py

# Run Streaming Gradio web interface with Docker
run-gradio-streaming-docker:
	@echo "Running Streaming Gradio web interface with Docker..."
	mkdir -p $(PWD)/output/gradio
	docker run -it --rm \
	    --gpus all \
	    -p 7860:7860 -p 7861:7861 \
	    -v $(PWD):/app \
	    -v $(PWD)/ckpts:/app/ckpts \
	    -v $(PWD)/downloaded_models:/app/downloaded_models \
	    -v $(PWD)/output:/app/output \
	    -e USE_HF_MODELS=0 \
	    -e PYTHONPATH=/app \
	    hymotion:latest python gradio_app_streaming.py

# Download and setup streaming-specific models
setup-streaming:
	@echo "Setting up streaming environment..."
	mkdir -p ckpts/tencent/HY-Motion-1.0
	mkdir -p downloaded_models/HY-Motion-1.0-Lite
	@echo "Checkpoint directories created"

# Fix checkpoint paths for streaming
fix-checkpoint-paths:
	@echo "Fixing checkpoint paths for streaming..."
	# Create symlink if Lite model exists in downloaded_models
	if [ -d "downloaded_models/HY-Motion-1.0-Lite" ]; then \
	    ln -sf ../downloaded_models/HY-Motion-1.0-Lite ckpts/tencent/HY-Motion-1.0 || true; \
	fi
	# Create symlink if full model exists in downloaded_models
	if [ -d "downloaded_models/HY-Motion-1.0" ]; then \
	    ln -sf ../downloaded_models/HY-Motion-1.0 ckpts/tencent/HY-Motion-1.0 || true; \
	fi
	@echo "Checkpoint path fixes applied"

# Complete streaming setup (download + fix paths)
streaming-setup: download-t2m fix-checkpoint-paths

# Test Docker dependencies
test-docker-deps:
	@echo "Testing Docker dependencies..."
	python test_docker_dependencies.py

# Test Docker dependencies in container
test-docker-deps-in-container:
	@echo "Testing dependencies in Docker container..."
	docker run -it --rm \\
	    --gpus all \\
	    -v $(pwd):/app \\
	    hymotion:latest \\
	    python test_docker_dependencies.py

# Clean up temporary files
clean:
	@echo "Cleaning up temporary files..."
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache .mypy_cache
	rm -rf downloaded_models
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*.pyd" -delete
	find . -name "*.pyo" -delete
	find . -name "*.pyd" -delete

# Build Docker image
docker-build:
	@echo "Building Docker image..."
	docker build -t hymotion:latest .

# Run Docker container
docker-run:
	@echo "Running Docker container..."
	mkdir -p $(PWD)/output/gradio
	docker run -it --rm \
	    --gpus all \
	    -p 7860:7860 \
	    -v $(PWD):/app \
	    -v $(PWD)/ckpts:/app/ckpts \
	    -v $(PWD)/downloaded_models:/app/downloaded_models \
	    -v $(PWD)/output:/app/output \
	    -e USE_HF_MODELS=0 \
	    -e PYTHONPATH=/app \
	    hymotion:latest bash -c "python --version && python3 --version && bash"

# Run Docker container with Python shell
docker-python:
	@echo "Running Docker container with Python shell..."
	mkdir -p $(PWD)/output/gradio
	docker run -it --rm \
	    --gpus all \
	    -p 7860:7860 \
	    -v $(PWD):/app \
	    -v $(PWD)/ckpts:/app/ckpts \
	    -v $(PWD)/downloaded_models:/app/downloaded_models \
	    -v $(PWD)/output:/app/output \
	    -e USE_HF_MODELS=0 \
	    -e PYTHONPATH=/app \
	    hymotion:latest python

# Run Docker container with Python3 shell
docker-python3:
	@echo "Running Docker container with Python3 shell..."
	mkdir -p $(PWD)/output/gradio
	docker run -it --rm \
	    --gpus all \
	    -p 7860:7860 \
	    -v $(PWD):/app \
	    -v $(PWD)/ckpts:/app/ckpts \
	    -v $(PWD)/downloaded_models:/app/downloaded_models \
	    -v $(PWD)/output:/app/output \
	    -e USE_HF_MODELS=0 \
	    -e PYTHONPATH=/app \
	    hymotion:latest python3

# Additional convenience targets
run-cli-with-rewrite:
	@echo "Running CLI inference with prompt rewriting enabled..."
	USE_HF_MODELS=0 python3 local_infer.py --model_path ckpts/tencent/HY-Motion-1.0 --disable_duration_est false --disable_rewrite false

run-cli-no-rewrite:
	@echo "Running CLI inference with prompt rewriting disabled..."
	USE_HF_MODELS=0 python3 local_infer.py --model_path ckpts/tencent/HY-Motion-1.0 --disable_duration_est true --disable_rewrite true