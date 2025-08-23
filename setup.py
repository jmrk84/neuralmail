#!/usr/bin/env python3
"""
Setup script for NeuralMail - AI-powered email search and analysis tool.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="neuralmail",
    version="1.0.0",
    author="NeuralMail Team",
    author_email="contact@neuralmail.app",
    description="AI-powered email search and analysis tool with multi-account support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/neuralmail",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/neuralmail/issues",
        "Documentation": "https://github.com/yourusername/neuralmail/blob/main/README.md",
        "Source Code": "https://github.com/yourusername/neuralmail",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Communications :: Email",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Environment :: X11 Applications :: Qt",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        'neuralmail': ['icon.ico'],
    },
    entry_points={
        'console_scripts': [
            'neuralmail=neuralmail.main:main',
        ],
    },
    keywords="email, ai, search, analysis, imap, nlp, rag, productivity",
    zip_safe=False,  # Required for PyQt5 applications
)
