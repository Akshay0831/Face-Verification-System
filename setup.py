"""
Setup script for the Modular Face Verification System
"""

from setuptools import setup, find_packages
import os

# Read the requirements file
def read_requirements():
    requirements = []
    with open('requirements.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('-'):
                requirements.append(line)
    return requirements

# Read the README file if it exists
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

setup(
    name="face_verification_system",
    version="1.0.0",
    author="Face Verification System Team",
    description="A modular, plugin-based face verification system using pre-2023 technologies",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/faceverification/modular-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Security",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.1",
            "mock>=4.0.3",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "mobile": [
            "Kivy>=2.1.0",
            "buildozer>=1.4.1",
        ],
        "desktop": [
            "PyQt5>=5.15.6",
            "Tkinter",
        ],
        "web": [
            "Flask>=2.0.1",
            "Flask-SocketIO>=5.1.1",
            "requests>=2.26.0",
        ],
        "gpu": [
            "tensorflow-gpu>=2.6.0",
        ],
        "security": [
            "cryptography>=3.4.7",
            "bcrypt>=3.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "face-verifier=face_verification_system.cli:main",
            "face-server=face_verification_system.server:main",
            "face-train=face_verification_system.training:main",
        ],
    },
    include_package_data=True,
    package_data={
        "face_verification_system": [
            "config/*.yaml",
            "config/*.json",
            "models/*.h5",
            "models/*.xml",
            "data/**/*",
        ],
    },
    keywords="face recognition, biometric, security, computer vision, neural networks, tensorflow",
    project_urls={
        "Bug Reports": "https://github.com/faceverification/modular-system/issues",
        "Source": "https://github.com/faceverification/modular-system",
        "Documentation": "https://faceverification.readthedocs.io/",
    },
)