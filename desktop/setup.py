"""Setup script for desktop interface"""

from setuptools import setup, find_packages

with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="face-verification-desktop",
    version="1.0.0",
    author="Face Verification System Team",
    author_email="team@faceverification.com",
    description="Desktop interface for Face Verification System",
    long_description="""
    A PyQt5-based desktop application for face verification tasks.
    Provides comprehensive interface for face detection, recognition, and liveness checking.
    
    Features:
    - Multi-algorithm face detection
    - Face recognition with deep learning
    - Liveness detection
    - User management
    - System monitoring
    - Configurable settings
    """,
    long_description_content_type="text/markdown",
    url="https://github.com/example/face-verification-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "face-verification-desktop=desktop.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)