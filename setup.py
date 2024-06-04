from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="picolearn",
    version="3.2.0",
    description="A simpler library for the alapaca trade api",
    url="https://github.com/AxelGard/picolearn",
    author="Axel Gard",
    author_email="axel.gard@tutanota.com",
    license="MIT",
    packages=["picolearn"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy==1.26.4",
    ],
    extras_requires={
        "dev": [
            "pytest",
            "sklearn",
            "black"
        ]
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
    ],
)
