from setuptools import setup, find_packages

base_packages = ["scikit-learn>=1.0.0", "polars>=0.20.10", "ucimlrepo>=0.0.3"]

setup(
    name="benchy",
    version="0.0.1",
    packages=find_packages(exclude=["notebooks"]),
    install_requires=base_packages,
)