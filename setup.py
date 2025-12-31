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
    version="1.0.0",
    description="HY-Motion 1.0: Scaling Flow Matching Models for 3D Motion Generation",
    author="Tencent Hunyuan 3D Digital Human Team",
    author_email="",
    url="https://github.com/Tencent-Hunyuan/HY-Motion-1.0",
    packages=find_packages(),
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
        "": ["*.yml", "*.yaml", "*.json", "*.txt"],
    },
)