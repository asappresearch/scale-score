from io import open

from setuptools import find_packages, setup
from pathlib import Path


REQUIREMENTS_PATH = Path(__file__).resolve().parent / "requirements.txt"

with open(str(REQUIREMENTS_PATH), "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="scale-score",
    version="0.0.0",
    author="Barrett Lattimer, Patrick Chen, Xinyuan Zhang, Yi Yang",
    author_email="blattimer@asapp.com",
    description="Implementation of SCALE metric and ScreenEval",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="Hallucination detection, Inconsistency detection, hallucination, automatic evaluation, metric, long document, efficient, fast, accurate, long, natural language generation, task agnostic, nlp, nlg",
    license="MIT",
    url="https://github.com/asappresearch/scale-score",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=requirements,
    include_package_data=True,
    python_requires=">=3.6",
    tests_require=["pytest"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_data={"scale_score": ["py.typed"]},
)
