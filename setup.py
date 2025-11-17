"""
setup.py for chaosnet (future_repo_crap)
---------------------------------
Why this exists
- Provides packaging metadata and dependencies so the copied package can be
  installed and imported elsewhere.

How it works
- Uses setuptools to discover packages and declare runtime/dev/doc extras.
  Adjust metadata (name, author, url) before distribution.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="chaosnet",
    version="0.1.0",
    author="Likara789",
    author_email="lowkeytripping.dev@gmail.com",
    description="A robust and efficient neural network architecture inspired by chaotic systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Likara789/chaosnet",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "black>=22.3.0",
            "isort>=5.10.1",
            "flake8>=4.0.1",
            "mypy>=0.942",
            "pytest>=7.1.2",
            "pytest-cov>=3.0.0",
            "pre-commit>=2.19.0",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
    },
)
