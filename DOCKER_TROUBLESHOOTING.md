# Docker Troubleshooting Guide for HY-Motion

## 🚨 Common Issues & Solutions

### 1. **Checkpoint Not Found Errors**

#### **Symptom**:
```
Failed to initialize runtime: [Errno 2] No such file or directory: 'ckpts/tencent/HY-Motion-1.0/config.yml'
```

#### **Solutions**:

**Option A: Volume Mount (Recommended)**
```bash
# Make sure checkpoints exist on host
docker run -it --rm \
    --gpus all \
    -v $(pwd)/ckpts:/app/ckpts \
    -p 7860:7860 \
    hymotion:latest python gradio_app_streaming.py
```

**Option B: Copy Checkpoints to Container**
```bash
# Copy checkpoints to container
docker cp ckpts/tencent/HY-Motion-1.0 container_name:/app/ckpts/tencent/
```

**Option C: Download in Container**
```bash
# Let the app download from Hugging Face
docker run -it --rm \
    --gpus all \
    -e USE_HF_MODELS=1 \
    -p 7860:7860 \
    hymotion:latest python gradio_app_streaming.py
```

**Option D: Use Docker Setup Script**
```bash
./docker_setup.sh
# Choose option 4 to run with automatic checkpoint handling
```

### 2. **Animation Not Building**

#### **Symptom**:
- App runs but no animation is generated
- FBX files are missing
- Visualization fails

#### **Solutions**:

**Check FBX Module**
```bash
python -c "try: import fbx; print('✅ FBX available')
except: print('❌ FBX missing')"
```

**Install FBX Module**
```bash
pip install fbxsdkpy==2020.1.post2
```

**Check Body Model**
```bash
python -c "
from hymotion.pipeline.motion_diffusion import MotionFlowMatching
print('✅ Body model available')
"
```

**Test Animation Generation**
```bash
python -c "
from hymotion.utils.t2m_runtime import T2MRuntime
runtime = T2MRuntime(
    config_path='ckpts/tencent/HY-Motion-1.0/config.yml',
    ckpt_name='ckpts/tencent/HY-Motion-1.0/latest.ckpt',
    skip_model_loading=True
)
print('✅ Animation system ready')
"
```

### 3. **Port Already in Use**

#### **Symptom**:
```
OSError: Cannot find empty port in range: 7860-7860
```

#### **Solutions**:

**Use Different Port**
```bash
python gradio_app_streaming.py --port 7861
```

**Kill Existing Process**
```bash
# Find and kill process on port 7860
lsof -ti:7860 | xargs kill -9
```

**Use Automatic Port Selection**
```bash
# The app automatically finds available ports
python gradio_app_streaming.py
```

### 4. **CUDA/GPU Issues**

#### **Symptom**:
- GPU not detected
- CUDA errors
- Slow performance

#### **Solutions**:

**Check GPU Availability**
```bash
nvidia-smi
```

**Test CUDA in Container**
```bash
python -c "import torch; print('✅ CUDA available' if torch.cuda.is_available() else '❌ CUDA not available')"
```

**Use CPU Fallback**
```bash
export HY_MOTION_DEVICE=cpu
python gradio_app_streaming.py
```

### 5. **Dependency Issues**

#### **Symptom**:
- Module not found errors
- Version conflicts
- Import errors

#### **Solutions**:

**Reinstall Dependencies**
```bash
pip install -r requirements.txt
```

**Check Specific Packages**
```bash
pip show gradio torch huggingface_hub
```

**Use Docker Build Cache**
```bash
docker build --no-cache -t hymotion:latest .
```

## 🐳 Docker-Specific Solutions

### **Docker Volume Permissions**

```bash
# Fix permission issues
docker run -it --rm \
    --user $(id -u):$(id -g) \
    -v $(pwd):/app \
    hymotion:latest bash
```

### **Docker Network Issues**

```bash
# Test network connectivity in container
docker run -it --rm hymotion:latest ping -c 4 google.com
```

### **Docker Resource Limits**

```bash
# Increase Docker resources
docker run -it --rm \
    --memory="16g" \
    --cpus="4" \
    --gpus all \
    hymotion:latest python gradio_app_streaming.py
```

## 🔍 Debugging Commands

### **Check Container Logs**
```bash
docker logs container_name
docker logs --follow container_name
```

### **Inspect Container**
```bash
docker inspect container_name
docker exec -it container_name bash
```

### **Test Checkpoint Paths**
```bash
docker exec container_name ls -la /app/ckpts/tencent/HY-Motion-1.0/
```

## 🎯 Recommended Workflow

### **1. Build Image**
```bash
docker build -t hymotion:latest .
```

### **2. Prepare Checkpoints**
```bash
# Option A: Download
make download-t2m

# Option B: Use existing
mkdir -p ckpts/tencent/HY-Motion-1.0
# Copy config.yml and latest.ckpt
```

### **3. Run with Proper Mounts**
```bash
docker run -it --rm \
    --gpus all \
    -v $(pwd)/ckpts:/app/ckpts \
    -v $(pwd):/app \
    -p 7860:7860 \
    -e PYTHONPATH=/app \
    -e USE_HF_MODELS=0 \
    hymotion:latest \
    python gradio_app_streaming.py
```

### **4. Troubleshoot if Needed**
```bash
./docker_setup.sh  # Use the interactive troubleshooter
```

## 📋 Checklist for Successful Setup

- [ ] Docker installed and running
- [ ] NVIDIA Container Toolkit installed (for GPU)
- [ ] Checkpoints available (local or will download)
- [ ] Proper volume mounts configured
- [ ] Ports 7860-7861 available
- [ ] Sufficient GPU memory (8GB+ recommended)
- [ ] All dependencies installed
- [ ] FBX module available (for animation)

## 🚀 Quick Fixes

### **One-Line Solution**
```bash
./docker_setup.sh && make streaming-setup && make run-gradio-streaming-docker
```

### **Minimal Working Example**
```bash
# Build
docker build -t hymotion:latest .

# Run with auto-download
docker run -it --rm --gpus all -p 7860:7860 hymotion:latest \
    python gradio_app_streaming.py
```

### **With Local Checkpoints**
```bash
# Build
docker build -t hymotion:latest .

# Run with local checkpoints
docker run -it --rm --gpus all \
    -v $(pwd)/ckpts:/app/ckpts \
    -p 7860:7860 \
    hymotion:latest python gradio_app_streaming.py
```

## 🎓 Advanced Troubleshooting

### **Check Model Loading**
```bash
python -c "
from hymotion.utils.t2m_runtime import T2MRuntime
runtime = T2MRuntime(
    config_path='ckpts/tencent/HY-Motion-1.0/config.yml',
    ckpt_name='ckpts/tencent/HY-Motion-1.0/latest.ckpt',
    skip_model_loading=False
)
print('Model loaded successfully')
"
```

### **Test Motion Generation**
```bash
python -c "
from hymotion.utils.t2m_runtime import T2MRuntime
runtime = T2MRuntime(
    config_path='ckpts/tencent/HY-Motion-1.0/config.yml',
    ckpt_name='ckpts/tencent/HY-Motion-1.0/latest.ckpt'
)
result = runtime.generate_motion('A person walking', '42', 3.0, 7.0)
print('Motion generation successful')
"
```

### **Check Animation Export**
```bash
python -c "
from hymotion.utils.t2m_runtime import T2MRuntime
runtime = T2MRuntime(
    config_path='ckpts/tencent/HY-Motion-1.0/config.yml',
    ckpt_name='ckpts/tencent/HY-Motion-1.0/latest.ckpt'
)
print('FBX available:', runtime.fbx_available)
"
```

## 📚 Additional Resources

- **Docker Documentation**: https://docs.docker.com/
- **NVIDIA Container Toolkit**: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/
- **HY-Motion GitHub**: https://github.com/your-repo/hy-motion
- **Gradio Documentation**: https://gradio.app/

This guide should help you resolve the "never builds the animation" issue and get HY-Motion working properly in Docker! 🎬