#!/usr/bin/env python3
"""
Setup script for CUDAgent - AI-Powered CUDA Kernel Optimization
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cudagent",
    version="0.1.0",
    author="CUDAgent Team",
    author_email="contact@cudagent.ai",
    description="AI-Powered CUDA Kernel Optimization for GPU Engineers",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/cudagent/cudagent",
    project_urls={
        "Bug Tracker": "https://github.com/cudagent/cudagent/issues",
        "Documentation": "https://docs.cudagent.ai",
        "Source Code": "https://github.com/cudagent/cudagent",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Hardware :: Symmetric Multi-processing",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.0",
        ],
        "gpu": [
            "torch>=1.9.0",
            "torchvision>=0.10.0",
        ],
        "llm": [
            "openai>=1.0.0",
            "anthropic>=0.5.0",
        ],
        "plotting": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cudagent=cudagent.cli:main",
            "cudagent-setup=cudagent.cli.setup:main",
            "cudagent-test=cudagent.cli.test:main",
            "cudagent-optimize=cudagent.cli.optimize:main",
            "cudagent-benchmark=cudagent.cli.benchmark:main",
        ],
    },
    include_package_data=True,
    package_data={
        "cudagent": [
            "config/*.json",
            "examples/*.py",
            "templates/*.cu",
        ],
    },
    keywords=[
        "cuda",
        "gpu",
        "optimization",
        "ai",
        "llm",
        "pytorch",
        "machine-learning",
        "deep-learning",
        "performance",
        "kernel",
    ],
    license="MIT",
    zip_safe=False,
) 