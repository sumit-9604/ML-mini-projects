"""
Setup configuration for NYC Taxi Trip Duration Prediction package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nyc-taxi-trip-duration",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Machine learning pipeline for predicting NYC taxi trip duration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nyc-taxi-trip-duration",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Development Status :: 4 - Beta",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "taxi-predict=scripts.predict:main",
            "taxi-train=scripts.train:main",
        ]
    },
)
