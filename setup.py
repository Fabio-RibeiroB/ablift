from setuptools import setup, find_packages

setup(
    name="bayestest",
    version="0.1.0",
    author="Fabio-RibeiroB",
    author_email="bradyfr@proton.me",
    description="A library to simplify Bayesian A/B/n testing",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bayestest",
    packages=find_packages(),
    install_requires=[
        "pymc>=4.0",  # Specify your dependencies here
        "numpy",
        "scipy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
