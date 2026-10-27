from setuptools import setup

setup(
    name="pysplitlock",
    version="0.1.0",
    author="Venkata Sai Swathi Gattamaneni",
    description="Static Partition-Taint Analysis and Metamorphic Patch Rejection for Autonomous ML Agents",
    py_modules=["pysplitlock"],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.22.0",
        "scikit-learn>=1.1.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)