#!/usr/bin/env python3
"""
Test script to verify Docker dependencies are properly installed.
This can be run inside the Docker container to check if everything is set up correctly.
"""

import sys
import subprocess

def test_dependencies():
    """Test that all required dependencies are available."""
    
    print("🔍 Testing HY-Motion Docker Dependencies")
    print("=" * 50)
    
    # Test basic Python packages
    basic_packages = [
        ('numpy', 'import numpy'),
        ('scipy', 'import scipy'),
        ('torch', 'import torch'),
        ('gradio', 'import gradio'),
        ('huggingface_hub', 'import huggingface_hub'),
        ('PyYAML', 'import yaml'),
    ]
    
    print("\n📦 Testing Basic Packages:")
    all_ok = True
    
    for name, import_stmt in basic_packages:
        try:
            exec(import_stmt)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            all_ok = False
    
    # Test HY-Motion specific imports
    print("\n🔧 Testing HY-Motion Modules:")
    hymotion_modules = [
        ('T2MRuntime', 'from hymotion.utils.t2m_runtime import T2MRuntime'),
        ('MotionFlowMatching', 'from hymotion.pipeline.motion_diffusion import MotionFlowMatching'),
        ('visualize_mesh_web', 'from hymotion.utils.visualize_mesh_web import generate_static_html_content'),
    ]
    
    for name, import_stmt in hymotion_modules:
        try:
            exec(import_stmt)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            all_ok = False
    
    # Test CUDA availability
    print("\n💻 Testing CUDA Availability:")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"{'✅ CUDA available' if cuda_available else '❌ CUDA not available'}")
        if cuda_available:
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
    except Exception as e:
        print(f"❌ CUDA check failed: {e}")
        all_ok = False
    
    # Test Gradio version
    print("\n🎨 Testing Gradio:")
    try:
        import gradio
        print(f"✅ Gradio {gradio.__version__}")
    except ImportError as e:
        print(f"❌ Gradio: {e}")
        all_ok = False
    
    # Test FBX module (optional)
    print("\n🎬 Testing FBX Module (optional):")
    try:
        import fbx
        print("✅ FBX module available")
    except ImportError:
        print("⚠️  FBX module not available (animations will be disabled)")
    
    print("\n" + "=" * 50)
    if all_ok:
        print("🎉 All dependencies are properly installed!")
        print("✅ Docker container is ready to use")
        return 0
    else:
        print("❌ Some dependencies are missing")
        print("✅ Please check the error messages above")
        return 1

if __name__ == "__main__":
    sys.exit(test_dependencies())