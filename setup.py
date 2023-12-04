"""A repository for experiments in Lucca."""

from setuptools import setup, find_packages

setup(
    name="imitate",
    version="0.1",
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        "numpy",
        "pandas",
        "casadi",
        "scipy",
        "matplotlib",
        "torch",
        "gym",
        "tqdm",
        "black",
        "pre-commit",
        "ipykernel",
        "seaborn",
    ],
)
