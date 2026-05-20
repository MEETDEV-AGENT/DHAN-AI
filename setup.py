"""Setup script for the Dhan AI package."""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", encoding="utf-8") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="dhan-ai",
    version="0.1.0",
    author="Meet Malpani",
    description=(
        "Intelligent Indian Stock Market Agent — "
        "Multi-agent orchestration for financial market intelligence"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MEETDEV-AGENT/DHAN-AI",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
