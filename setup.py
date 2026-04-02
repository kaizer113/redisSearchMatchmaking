from setuptools import find_packages, setup


setup(
    name="matchmaking-data",
    version="0.1.0",
    description="Synthetic player dataset generator and Redis loader for matchmaking vector search",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "einops>=0.8.0",
        "numpy>=1.24.0",
        "redis>=5.0.0",
        "sentence-transformers>=3.0.0",
        "torch>=2.0.0",
    ],
)
