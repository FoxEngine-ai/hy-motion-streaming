# Gradio Streaming Troubleshooting Guide

## 🚨 Issue: "🕒 Generating motion frames..." Stuck

### **Symptom**
- CLI generates frames successfully
- Gradio interface shows "🕒 Generating motion frames..." but never completes
- No animation is generated
- No error messages visible

### **Root Causes & Solutions**

## 🔍 **1. Runtime Not Initialized**

### **Check**
```bash
# Look for this message in console:
"WARNING: Runtime not initialized. Streaming generation will fail."
```

### **Solutions**

**A. Check Checkpoint Availability**
```bash
ls -la ckpts/tencent/HY-Motion-1.0/
# Should show: config.yml and latest.ckpt
```

**B. Download Checkpoints**
```bash
make download-t2m
```

**C. Use Volume Mount (Docker)**
```bash
docker run -v $(pwd)/ckpts:/app/ckpts hymotion:latest python gradio_app_streaming.py
```

**D. Check Runtime Initialization**
```python
# Test runtime initialization
python -c "
from hymotion.utils.t2m_runtime import T2MRuntime
try:
    runtime = T2MRuntime(
        config_path='ckpts/tencent/HY-Motion-1.0/config.yml',
        ckpt_name='ckpts/tencent/HY-Motion-1.0/latest.ckpt',
        skip_model_loading=False
    )
    print('✅ Runtime initialized successfully')
except Exception as e:
    print(f'❌ Runtime initialization failed: {e}')
    import traceback
    traceback.print_exc()
"
```

## 🔄 **2. Frame Generation Stuck**

### **Check**
```bash
# Look for frame generation messages:
"Frame X/Y generated (progress: Z%)"
```

### **Solutions**

**A. Check Console Output**
```bash
# Run with debug output
python gradio_app_streaming.py 2>&1 | grep "Frame"
```

**B. Test Frame Generation Directly**
```python
# Test frame generation
python -c "
from hymotion.utils.t2m_runtime import T2MRuntime
runtime = T2MRuntime(
    config_path='ckpts/tencent/HY-Motion-1.0/config.yml',
    ckpt_name='ckpts/tencent/HY-Motion-1.0/latest.ckpt'
)

# Test streaming generation
frame_generator = runtime.generate_motion_streaming(
    text='A person walking',
    seeds_csv='42',
    duration=3.0,
    cfg_scale=7.0
)

for i, frame_data in enumerate(frame_generator):
    print(f'Frame {i}: {frame_data["progress"]:.1%}')
    if i > 5:  # Test first few frames
        break
"
```

**C. Check for Deadlocks**
```bash
# Look for stuck threads
python -c "
import threading
print(f'Active threads: {threading.active_count()}')
for thread in threading.enumerate():
    print(f'{thread.name}: {thread.ident}')
"
```

## 📁 **3. HTML Generation Failed**

### **Check**
```bash
# Look for HTML generation messages:
"HTML generation successful" or "HTML generation failed"
```

### **Solutions**

**A. Check Output Directory**
```bash
mkdir -p output/gradio
chmod 777 output/gradio
```

**B. Test HTML Generation**
```python
# Test HTML generation
python -c "
from hymotion.utils.t2m_runtime import T2MRuntime
runtime = T2MRuntime(
    config_path='ckpts/tencent/HY-Motion-1.0/config.yml',
    ckpt_name='ckpts/tencent/HY-Motion-1.0/latest.ckpt'
)

try:
    html = runtime._generate_html_content(
        timestamp='test',
        file_path='test',
        output_dir='output/gradio'
    )
    print(f'✅ HTML generation successful, length: {len(html)}')
except Exception as e:
    print(f'❌ HTML generation failed: {e}')
    import traceback
    traceback.print_exc()
"
```

**C. Check Dependencies**
```bash
# Check required dependencies
python -c "
try:
    import numpy as np
    import torch
    from hymotion.utils.visualize_mesh_web import generate_static_html_content
    print('✅ All dependencies available')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
"
```

## 🎯 **4. Gradio Interface Issues**

### **Check**
```bash
# Check Gradio version
python -c "import gradio; print(f'Gradio version: {gradio.__version__}')"
```

### **Solutions**

**A. Update Gradio**
```bash
pip install --upgrade gradio==5.38.2
```

**B. Check Gradio Interface**
```python
# Test Gradio interface
python -c "
import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown('Test Interface')
    btn = gr.Button('Test')
    output = gr.Textbox()
    btn.click(fn=lambda: 'Working!', outputs=output)

demo.launch(server_name='0.0.0.0', server_port=7862, show_error=True)
print('✅ Gradio interface test started on port 7862')
"
```

**C. Check Event Handlers**
```python
# Test event handlers
python -c "
from gradio_app_streaming import StreamingGradioApp
app = StreamingGradioApp()
print(f'Runtime initialized: {app.runtime is not None}')
print(f'FBX available: {app.fbx_available}')
print(f'Prompt engineering: {app.prompt_engineering_available}')
"
```

## 🐳 **5. Docker-Specific Issues**

### **Check**
```bash
# Check Docker volume mounts
docker inspect container_name | grep -A 10 Mounts
```

### **Solutions**

**A. Verify Volume Mounts**
```bash
# Check if checkpoints are mounted
docker exec container_name ls -la /app/ckpts/tencent/HY-Motion-1.0/
```

**B. Check File Permissions**
```bash
# Fix permissions
docker exec container_name chmod -R 777 /app/ckpts
docker exec container_name chmod -R 777 /app/output
```

**C. Test in Container**
```bash
# Test runtime in container
docker exec -it container_name python -c "
from hymotion.utils.t2m_runtime import T2MRuntime
runtime = T2MRuntime(
    config_path='/app/ckpts/tencent/HY-Motion-1.0/config.yml',
    ckpt_name='/app/ckpts/tencent/HY-Motion-1.0/latest.ckpt'
)
print('✅ Runtime works in container')
"
```

## 🔧 **Debugging Steps**

### **Step 1: Check Runtime Initialization**
```bash
# Look for runtime initialization messages
python gradio_app_streaming.py 2>&1 | grep -E "(Initializing|Failed to initialize|Runtime)"
```

### **Step 2: Check Frame Generation**
```bash
# Look for frame generation progress
python gradio_app_streaming.py 2>&1 | grep -E "Frame [0-9]+/[0-9]+"
```

### **Step 3: Check HTML Generation**
```bash
# Look for HTML generation messages
python gradio_app_streaming.py 2>&1 | grep -E "(HTML generation|Static HTML)"
```

### **Step 4: Check Errors**
```bash
# Look for any error messages
python gradio_app_streaming.py 2>&1 | grep -E "(Error|Exception|Traceback|Failed)"
```

## ✅ **Solutions Summary**

### **Quick Fixes**

**1. Restart with Debug Output**
```bash
python gradio_app_streaming.py 2>&1 | tee streaming_debug.log
```

**2. Test Minimal Generation**
```bash
python -c "
from hymotion.utils.t2m_runtime import T2MRuntime
runtime = T2MRuntime(
    config_path='ckpts/tencent/HY-Motion-1.0/config.yml',
    ckpt_name='ckpts/tencent/HY-Motion-1.0/latest.ckpt'
)

# Test minimal generation
result = list(runtime.generate_motion_streaming(
    text='test',
    seeds_csv='42',
    duration=1.0,
    cfg_scale=7.0
))
print(f'✅ Generated {len(result)} frames')
"
```

**3. Use Docker Setup Script**
```bash
./docker_setup.sh
# Choose option 4 for streaming with automatic troubleshooting
```

### **Advanced Fixes**

**1. Manual Frame Generation Test**
```python
# Test frame-by-frame generation
python -c "
from hymotion.utils.t2m_runtime import T2MRuntime
import time

runtime = T2MRuntime(
    config_path='ckpts/tencent/HY-Motion-1.0/config.yml',
    ckpt_name='ckpts/tencent/HY-Motion-1.0/latest.ckpt'
)

frame_generator = runtime.generate_motion_streaming(
    text='A person walking',
    seeds_csv='42',
    duration=2.0,
    cfg_scale=7.0
)

for frame_data in frame_generator:
    print(f'Frame {frame_data["frame_index"]}/{frame_data["total_frames"]} - {frame_data["progress"]:.1%}')
    if frame_data.get('completed'):
        print('✅ Generation completed!')
        break
    time.sleep(0.1)
"
```

**2. Test HTML Generation Directly**
```python
# Test HTML generation with mock data
python -c "
from hymotion.utils.t2m_runtime import T2MRuntime
import os

# Create test output directory
os.makedirs('test_output', exist_ok=True)

runtime = T2MRuntime(
    config_path='ckpts/tencent/HY-Motion-1.0/config.yml',
    ckpt_name='ckpts/tencent/HY-Motion-1.0/latest.ckpt'
)

# Generate test motion first
result = runtime.generate_motion('test', '42', 1.0, 7.0, output_format='dict')

# Now test HTML generation
try:
    html = runtime._generate_html_content(
        timestamp='test',
        file_path='test_motion',
        output_dir='test_output'
    )
    print(f'✅ HTML generation successful: {len(html)} characters')
    with open('test_output.html', 'w') as f:
        f.write(html)
    print('✅ HTML saved to test_output.html')
except Exception as e:
    print(f'❌ HTML generation failed: {e}')
    import traceback
    traceback.print_exc()
"
```

## 🎯 **Common Error Patterns & Fixes**

### **Pattern 1: Runtime Not Initialized**
```
WARNING: Runtime not initialized. Streaming generation will fail.
```
**Fix**: Check checkpoints and runtime initialization

### **Pattern 2: Frame Generation Stuck**
```
Frame 0/50 generated (progress: 0.0%)
# No further frames
```
**Fix**: Check motion generation pipeline and CUDA availability

### **Pattern 3: HTML Generation Failed**
```
HTML generation failed: [error details]
```
**Fix**: Check output directory permissions and dependencies

### **Pattern 4: Silent Failure**
```
# No output at all
```
**Fix**: Check Gradio version and event handlers

## 📋 **Checklist for Successful Streaming**

- [ ] Runtime initialized successfully
- [ ] Checkpoints available at correct path
- [ ] Frame generation produces output
- [ ] HTML generation works
- [ ] Gradio interface responsive
- [ ] No JavaScript errors in browser console
- [ ] CUDA/GPU available (if using GPU)
- [ ] Sufficient memory (8GB+ recommended)
- [ ] Output directories writable
- [ ] All dependencies installed

## 🚀 **Recommended Workflow**

### **1. Test Runtime**
```bash
python -c "from hymotion.utils.t2m_runtime import T2MRuntime; T2MRuntime('ckpts/tencent/HY-Motion-1.0/config.yml', 'ckpts/tencent/HY-Motion-1.0/latest.ckpt'); print('✅ Runtime OK')"
```

### **2. Test Frame Generation**
```bash
python -c "from hymotion.utils.t2m_runtime import T2MRuntime; runtime = T2MRuntime('ckpts/tencent/HY-Motion-1.0/config.yml', 'ckpts/tencent/HY-Motion-1.0/latest.ckpt'); list(runtime.generate_motion_streaming('test', '42', 1.0, 7.0))[:3]; print('✅ Frames OK')"
```

### **3. Test HTML Generation**
```bash
python -c "from hymotion.utils.t2m_runtime import T2MRuntime; runtime = T2MRuntime('ckpts/tencent/HY-Motion-1.0/config.yml', 'ckpts/tencent/HY-Motion-1.0/latest.ckpt'); runtime._generate_html_content('test', 'test', 'output/gradio'); print('✅ HTML OK')"
```

### **4. Run Streaming App**
```bash
python gradio_app_streaming.py
```

### **5. Check Browser Console**
- Open browser developer tools (F12)
- Check for JavaScript errors
- Check network requests

## 📚 **Additional Resources**

- **Gradio Documentation**: https://gradio.app/
- **HY-Motion GitHub Issues**: https://github.com/your-repo/hy-motion/issues
- **Python Threading**: https://docs.python.org/3/library/threading.html
- **Docker Volumes**: https://docs.docker.com/storage/volumes/

This guide should help you resolve the "🕒 Generating motion frames..." issue and get the Gradio streaming interface working properly! 🎬