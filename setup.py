#!/usr/bin/env python3
"""
Setup script for IRONFORGE Archaeological Discovery System
"""

import os

from setuptools import find_packages, setup

# Read README if it exists
readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name="ironforge",
    version="1.0.0",
    author="IRON Ecosystem",
    author_email="noreply@iron.dev",
    description="Archaeological discovery system for market pattern analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iron-ecosystem/ironforge",
    packages=find_packages(include=['ironforge*']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "torch>=1.9.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "networkx>=2.5",
        "tqdm>=4.60.0",
        "PyYAML>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "ironforge=ironforge.sdk.cli:main",
            "ifg=ironforge.sdk.cli:main",
        ]
    },
)
