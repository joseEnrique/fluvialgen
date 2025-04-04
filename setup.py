from setuptools import setup, find_packages

setup(
    name="fluvialgen",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "geopandas",
        "shapely",
        "rasterio",
        "tqdm",
        "river",
    ],
    author="Jose Enrique Ruiz Navarro",
    author_email="joseenriqueruiznavarro@gmail.com",
    description="A Python package for generating synthetic river networks and datasets",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/joseenriqueruiznavarro/FluvialGen",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
) 