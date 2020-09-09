import setuptools

from abayes.__version__ import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="abayes",
    version=f"{__version__}",
    author="Raphael Sourty",
    author_email="raphael.sourty@gmail.com",
    description="Autoregressive bayesian linear model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/raphaelsty/abayes",
    packages=setuptools.find_packages(),
    package_data={
        'abayes': ['dataset/ice_cream.csv']
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6',
)
