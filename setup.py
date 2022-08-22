import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="omccolors",
    version="0.1.0",
    author="Daniel Braun",
    author_email="braun@cs.uni-koeln.de",
    description="Generation of Order of Magnitude Colors from Matplotlib Color Scales",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://arxiv.org/abs/2207.12399",
    license='MIT',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',

    install_requires=['numpy', 'matplotlib', 'colormath']
)