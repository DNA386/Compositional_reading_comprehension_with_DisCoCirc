from setuptools import setup, find_packages


setup(
    name="following_showcase",
    version="0.0.1",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch==2.0.1",
        "numpy<2",
        "discopy>=0.6,<1",
        "uvicorn>=0.30",
        "fastapi>=0.111",
    ]
)
