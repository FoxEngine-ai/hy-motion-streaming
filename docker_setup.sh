#!/bin/bash

# Comprehensive Docker Setup Script for HY-Motion
# Handles checkpoint mounting, animation generation, and troubleshooting

echo "🐳 HY-Motion Docker Setup Assistant"
echo "=================================="

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

echo "✅ Docker is running"

# Build the Docker image
build_image() {
    echo "🔨 Building HY-Motion Docker image..."
    docker build -t hymotion:latest -f Dockerfile .
    if [ $? -eq 0 ]; then
        echo "✅ Docker image built successfully"
    else
        echo "❌ Docker build failed"
        exit 1
    fi
}

# Check if checkpoints exist on host
check_host_checkpoints() {
    if [ -d "ckpts/tencent/HY-Motion-1.0" ]; then
        if [ -f "ckpts/tencent/HY-Motion-1.0/config.yml" ] && [ -f "ckpts/tencent/HY-Motion-1.0/latest.ckpt" ]; then
            echo "✅ Host checkpoints found and complete"
            return 0
        else
            echo "⚠️  Host checkpoint directory exists but incomplete"
            return 1
        fi
    else
        echo "❌ No host checkpoints found"
        return 1
    fi
}

# Download checkpoints if needed
download_checkpoints() {
    echo "📥 Downloading checkpoints..."
    make download-t2m
    if [ $? -eq 0 ]; then
        echo "✅ Checkpoints downloaded successfully"
    else
        echo "❌ Checkpoint download failed"
        return 1
    fi
}

# Run streaming app with proper checkpoint mounting
run_streaming() {
    echo "🚀 Starting HY-Motion Streaming App..."
    
    # Check for checkpoints
    if check_host_checkpoints; then
        echo "🔗 Mounting host checkpoints"
        VOLUME_ARGS="-v $(pwd)/ckpts:/app/ckpts"
    else
        echo "⚠️  No host checkpoints found, will use container checkpoints or download"
        VOLUME_ARGS=""
    fi
    
    # Run the container
    docker run -it --rm \\
        --gpus all \\
        -p 7860:7860 -p 7861:7861 \\
        -v $(pwd):/app \\
        $VOLUME_ARGS \\
        -e PYTHONPATH=/app \\
        -e USE_HF_MODELS=0 \\
        hymotion:latest \\
        python gradio_app_streaming.py
}

# Copy checkpoints to container (alternative approach)
copy_checkpoints_to_container() {
    echo "📦 Copying checkpoints to container..."
    
    # First create a temporary container
    CONTAINER_ID=$(docker create hymotion:latest)
    
    # Copy checkpoints
    if [ -d "ckpts/tencent/HY-Motion-1.0" ]; then
        docker cp ckpts/tencent/HY-Motion-1.0 $CONTAINER_ID:/app/ckpts/tencent/
        echo "✅ Checkpoints copied to container"
    else
        echo "❌ No checkpoints to copy"
        docker rm $CONTAINER_ID
        return 1
    fi
    
    # Commit the container with checkpoints
    docker commit $CONTAINER_ID hymotion-with-checkpoints:latest
    echo "✅ Created image with checkpoints: hymotion-with-checkpoints:latest"
    
    # Clean up
    docker rm $CONTAINER_ID
    
    # Run the new image
    docker run -it --rm \\
        --gpus all \\
        -p 7860:7860 -p 7861:7861 \\
        -v $(pwd):/app \\
        -e PYTHONPATH=/app \\
        -e USE_HF_MODELS=0 \\
        hymotion-with-checkpoints:latest \\
        python gradio_app_streaming.py
}

# Troubleshoot animation issues
troubleshoot_animation() {
    echo "🔍 Troubleshooting Animation Issues..."
    
    # Check common issues
    echo "Checking common animation issues:"
    
    # 1. Check if FBX module is available
    python -c "try: import fbx; print('✅ FBX module available')
    except ImportError: print('❌ FBX module missing')"
    
    # 2. Check if body model is available
    python -c "from hymotion.pipeline.motion_diffusion import MotionFlowMatching; print('✅ Body model available')"
    
    # 3. Check animation generation
    echo "Testing animation generation..."
    python -c "
from hymotion.utils.t2m_runtime import T2MRuntime
import os

# Test with a simple configuration
try:
    runtime = T2MRuntime(
        config_path='ckpts/tencent/HY-Motion-1.0/config.yml',
        ckpt_name='ckpts/tencent/HY-Motion-1.0/latest.ckpt',
        skip_model_loading=True,
        disable_prompt_engineering=True
    )
    print('✅ Runtime initialization successful')
except Exception as e:
    print(f'❌ Runtime initialization failed: {e}')
"
}

# Main menu
main_menu() {
    echo ""
    echo "🎯 HY-Motion Docker Setup Menu"
    echo "1. Build Docker image"
    echo "2. Check host checkpoints"
    echo "3. Download checkpoints"
    echo "4. Run streaming app (recommended)"
    echo "5. Copy checkpoints to container"
    echo "6. Troubleshoot animation issues"
    echo "7. Exit"
    echo ""
    
    read -p "Enter your choice (1-7): " choice
    
    case $choice in
        1) build_image ;;
        2) check_host_checkpoints ;;
        3) download_checkpoints ;;
        4) run_streaming ;;
        5) copy_checkpoints_to_container ;;
        6) troubleshoot_animation ;;
        7) echo "Goodbye! 👋"; exit 0 ;;
        *) echo "Invalid choice. Please try again."; main_menu ;;
    esac
}

# Start with main menu
main_menu