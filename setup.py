from setuptools import setup, find_packages

setup(
    name="autoconjecture",
    version="0.1.0",
    description="AI system for mathematical reasoning from scratch",
    author="AutoConjecture Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "plotly>=5.14.0",
            "streamlit>=1.28.0",
            "graphviz>=0.20.0",
        ],
    },
)
