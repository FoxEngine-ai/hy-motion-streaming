#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

# install_requires is a hand-curated, library-friendly subset of the full
# requirements.txt. Tencent's requirements.txt pins exact versions
# (torch==2.5.1, transformers==4.53.3, ...) for research reproducibility,
# which makes this package un-installable into environments that have
# already settled on different pins (e.g. Voxta on torch 2.8). For the
# runtime library we only declare the deps actually imported by code under
# `hymotion/` and use loose lower bounds; the host project pins the
# specific versions it wants.
#
# requirements.txt remains as a "tested environment" reference for the
# gradio demo and `local_infer.py` workflows.
INSTALL_REQUIRES = [
    "torch>=2.5.1",
    "transformers>=4.53.3",
    "diffusers>=0.26.3",
    "huggingface_hub>=0.30.0",
    "accelerate>=0.30.1",
    "torchdiffeq>=0.2.5",
    "einops>=0.8.1",
    "safetensors>=0.5.3",
    # No upper bound on numpy: Tencent's ceiling of <2.0 reflected an older
    # transformers/diffusers that hadn't migrated to numpy 2.x yet, but the
    # versions we require here (transformers >= 4.53.3, diffusers >= 0.26.3)
    # are numpy-2-compatible. Pinning <2.0 makes the wheel un-installable
    # alongside numpy-2 ecosystem packages (exllamav3, etc).
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "transforms3d>=0.4.2",
    "PyYAML>=6.0",
    "omegaconf>=2.3.0",
    "requests>=2.32.0",
]

# Optional feature sets — opted into by the user via `pip install hymotion[xxx]`.
# Each one carries a dependency that's only needed for a specific code path
# and would be wasteful (or break the pip resolver, or be platform-specific)
# to require unconditionally.
EXTRAS_REQUIRE = {
    # GGUF text-encoder path — loads Qwen3 as a quantized GGUF on CPU.
    "gguf": ["llama-cpp-python", "gguf>=0.10.0"],
    # bitsandbytes 4-bit / 8-bit quantization for the prompter and main
    # encoder. Linux + CUDA only on most distributions.
    "quantization": ["bitsandbytes>=0.49.0"],
    # OpenAI-backed prompt rewriter. Lazy-imported, so this is only needed
    # when callers actually use that feature.
    "openai": ["openai>=1.78.0"],
    # Gradio demo (gradio_app.py / gradio_app_streaming.py). Not needed for
    # library / streaming-server usage.
    "gradio": ["gradio==5.38.2", "click>=8.1.3"],
    # FBX export for animation pipelines.
    "fbx": ["fbxsdkpy==2020.1.post2"],
}
# `all` extra — everything optional in one go, for the gradio demo / dev setup.
EXTRAS_REQUIRE["all"] = sorted(set(p for v in EXTRAS_REQUIRE.values() for p in v))

setup(
    name="hymotion",
    version="1.1.2",
    description=(
        "HY-Motion 1.0 (FoxEngine fork): Scaling Flow Matching Models for 3D Motion "
        "Generation. Adds pip-installability, GGUF/quantization, and streaming on "
        "top of Tencent's research release."
    ),
    author="Tencent Hunyuan 3D Digital Human Team (upstream); FoxEngine.ai (fork)",
    author_email="",
    url="https://github.com/FoxEngine-ai/hy-motion-streaming",
    packages=find_packages(include=["hymotion", "hymotion.*"]),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
        "Topic :: Multimedia :: Graphics :: 3D Rendering",
    ],
    keywords="motion-generation text-to-motion 3d-animation diffusion-transformer flow-matching",
    include_package_data=True,
    package_data={
        # The .bin / .webp / .json / .ply files in hymotion/assets/dump_wooden/
        # back the WoodenMesh body-model loader; bundling them makes the wheel
        # self-contained for `pip install`. Generic *.yml / *.yaml / *.txt entry
        # preserves upstream behaviour for any other config files we add later.
        "": ["*.yml", "*.yaml", "*.json", "*.txt"],
        "hymotion.assets.dump_wooden": ["*.bin", "*.webp", "*.json", "*.ply"],
    },
)