from setuptools import setup, find_packages

setup(
    name="randomml",
    version="0.1.0",
    author="Your Name",
    author_email="your_email@example.com",
    description="An implementation of MLPedRVFL with Boosting and Bagging support.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yuvrajiro/random-ml",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "scikit-learn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
