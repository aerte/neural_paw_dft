# setup.py
#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="1.0",
    description="ELECTRAFI",
    python_requires=">=3.8",
    install_requires=[
        "lightning",
        "dill",
        "lz4",
        "ase",
        "torch_geometric",
        "plotly",
        "fairchem-core==1.10.0",
        "torch_scatter",
        "torch_cluster",
        "pykeops",
    ],
    packages=find_packages(),
)
