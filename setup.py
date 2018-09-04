import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyppca",
    version="0.0.3",
    author="Sheridan Green",
    author_email="sheridan.green@yale.edu",
    description="Probabilistic PCA with Missing Values",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shergreen/pyppca",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
     install_requires = [
      'numpy',
      'scipy',
    ],
)
