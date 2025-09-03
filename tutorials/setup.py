from setuptools import setup, find_packages


setup(
    name="discocirc_tutorials",
    version="0.0.1",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "lambeq>=0.5",
        "jupyter",
        "qutip",
        "pandas",
    ]
)
