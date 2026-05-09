#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = []
    for line in f:
        line = line.strip()
        # Skip empty lines and comments
        if line and not line.startswith('#'):
            # Skip the extra-index-url line as it's not a package requirement
            if not line.startswith('--extra-index-url'):
                requirements.append(line)

setup(
    name="hymotion",
    version="1.1.0",
    description=(
        "HY-Motion 1.0 (FoxEngine fork): Scaling Flow Matching Models for 3D Motion "
        "Generation. Adds pip-installability, GGUF/quantization, and streaming on "
        "top of Tencent's research release."
    ),
    author="Tencent Hunyuan 3D Digital Human Team (upstream); FoxEngine.ai (fork)",
    author_email="",
    url="https://github.com/FoxEngine-ai/hy-motion-streaming",
    packages=find_packages(include=["hymotion", "hymotion.*"]),
    install_requires=requirements,
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