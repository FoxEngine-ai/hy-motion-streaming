#!/bin/bash

# Script to rebuild the Docker image with updated requirements

echo "🐳 Rebuilding HY-Motion Docker image with Gradio support..."

echo "📋 Updated requirements.txt to include:"
echo "   - gradio==5.38.2"
echo ""

echo "🔧 Building Docker image..."
docker build -t hy-motion-streaming -f Dockerfile .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully!"
    echo ""
    echo "🚀 To run the streaming app:"
    echo "   docker run -it --gpus all \\"
    echo "     -v \"$PWD/ckpts:/app/ckpts\" \\"
    echo "     -p 7860:7860 -p 7861:7861 \\"
    echo "     hy-motion-streaming \\"
    echo "     python gradio_app_streaming.py"
else
    echo "❌ Docker build failed"
    exit 1
fi