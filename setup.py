from setuptools import find_packages, setup

setup(
    name="bayestest",
    version="0.3.0",
    author="Fabio-RibeiroB",
    author_email="bradyfr@proton.me",
    description="CLI and Python toolkit for Bayesian and sequential A/B test decisions",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Fabio-RibeiroB/bayestest",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.22",
        "openpyxl>=3.1.0",
    ],
    entry_points={
        "console_scripts": [
            "bayestest=bayestest.cli:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
