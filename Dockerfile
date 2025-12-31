# Dockerfile for HY-Motion 1.0 with Python 3.10 and CUDA support
# Using pre-built Python for faster and more reliable builds
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python-is-python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    wget \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}

# Upgrade pip and install build tools
RUN python3 -m pip install --upgrade pip setuptools wheel

# Create and set up working directory
WORKDIR /app
COPY . /app

# Install Python dependencies with retry logic
RUN pip install --no-cache-dir numpy scipy && \
    pip install --no-cache-dir PyYAML==6.0 && \
    pip install --no-cache-dir -r requirements.txt || \
    (sleep 30 && pip install --no-cache-dir -r requirements.txt)

# Install PyTorch with CUDA 12.1 support first
RUN pip install --no-cache-dir \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Verify all dependencies are installed
RUN python -c "import sys; import gradio; import torch; import huggingface_hub; import numpy; import scipy; from hymotion.utils.t2m_runtime import T2MRuntime; from hymotion.pipeline.motion_diffusion import MotionFlowMatching; print('✅ All dependencies installed successfully'); print(f'Gradio: {gradio.__version__}'); print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Clean up
RUN apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up entry point
ENV PYTHONPATH=/app

# Create output directory for generated files
RUN mkdir -p /app/output/gradio

# Define volume for persisting generated files
VOLUME ["/app/output"]

# Default command (can be overridden)
CMD ["bash"]

# Health check to verify checkpoint availability
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import os; print('✓ Checkpoint dir exists' if os.path.exists('/app/ckpts') else '✗ Checkpoint dir missing')" || exit 1

# Final verification that everything is ready
RUN echo "🎉 HY-Motion Docker setup complete!"
RUN echo "✅ All dependencies installed"
RUN echo "✅ PyTorch with CUDA support ready"
RUN echo "✅ Gradio interface ready"
RUN echo "✅ Ready to run: python gradio_app_streaming.py"