#!/usr/bin/env python3
"""
Iron-Core Package Setup
=======================
Shared infrastructure package for IRON ecosystem with dependency injection,
lazy loading, and mathematical component management.

Features:
- Lazy loading system (88.7% performance improvement)
- Thread-safe dependency injection containers
- Mathematical component validation
- Cross-suite integration framework
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read the README file for long description
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Shared infrastructure for IRON ecosystem"

# Version information
VERSION = "1.0.0"
AUTHOR = "IRON Ecosystem"
AUTHOR_EMAIL = "development@iron-ecosystem.com"

setup(
    name="iron-core",
    version=VERSION,
    description="Shared infrastructure for IRON ecosystem",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url="https://github.com/iron-ecosystem/iron-core",
    project_urls={
        "Documentation": "https://github.com/iron-ecosystem/iron-core/docs",
        "Source": "https://github.com/iron-ecosystem/iron-core",
        "Tracker": "https://github.com/iron-ecosystem/iron-core/issues",
    },
    
    # Package configuration
    packages=find_packages(),
    package_data={
        "iron_core": ["py.typed"],  # PEP 561 type information
    },
    include_package_data=True,
    
    # Core dependencies for mathematical computing and performance
    install_requires=[
        "torch>=1.9.0,<2.5.0",  # PyTorch for mathematical operations
        "numpy>=1.21.0,<2.0.0",  # Numerical computing foundation
        "scikit-learn>=1.0.0,<1.5.0",  # Machine learning utilities
        "typing-extensions>=4.0.0",  # Advanced typing support for Python <3.10
    ],
    
    # Optional dependencies for development and testing
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio>=0.18.0",
            "pytest-cov>=2.12.0",
            "black>=21.0",
            "mypy>=0.910",
        ],
        "performance": [
            "psutil>=5.8.0",  # System performance monitoring
            "memory-profiler>=0.60.0",  # Memory usage profiling
        ],
        "visualization": [
            "matplotlib>=3.5.0",  # Mathematical plotting
            "seaborn>=0.11.0",  # Statistical visualization
        ],
    },
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Entry points for command-line tools
    entry_points={
        "console_scripts": [
            "iron-core-test=iron_core.validation:test_installation",
        ],
    },
    
    # Package classifiers for PyPI
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry", 
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
    ],
    
    # Keywords for package discovery
    keywords=[
        "iron-core",
        "dependency-injection", 
        "lazy-loading",
        "performance-optimization",
        "mathematical-computing",
        "financial-modeling",
        "pytorch",
        "ecosystem-infrastructure",
    ],
    
    # Package metadata
    license="MIT",
    zip_safe=False,  # Required for type information
)