import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bioimg", # Replace with your own username
    version="0.0.1",
    author="Vladislav Kim",
    author_email="vladhkim@gmail.com",
    description="Single-cell analysis of microscopy data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vladchimescu/bioimg",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.3',
)
