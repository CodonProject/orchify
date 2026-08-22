from setuptools import setup, find_packages

setup(
    name="orchify",
    version="0.0.1a2",
    author="Orchify",
    description="A Python framework for building LLM-powered AI agent orchestration systems",
    package_dir={"": "."},
    packages=find_packages(where="."),
    python_requires=">=3.10",
    install_requires=[
        "httpx>=0.25.0",
        "python-dotenv>=1.0.0",
        "docstring-parser>=0.15",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
