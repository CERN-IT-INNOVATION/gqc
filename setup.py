from setuptools import setup, find_packages
from pathlib import Path
import sys

assert sys.version_info >= (3, 9), "This study requires python 3.9 or above."

CURRENT_DIR = Path(__file__).parent


def get_long_description() -> str:
    return (CURRENT_DIR / "README.md").read_text(encoding="utf8")


# Suggestion: Add entry points possibly.
# TODO: Add package data.
setup(
    name="qclassifier_study",
    author="Vasilis Belis, Patrick Odagiu, Samuel Gonzalez Castillo",
    author_email="podagiu@student.ethz.ch",
    version="1.0",
    description="Study on feature reduction used with quantum classifiers.",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    keywords="qml autoencoders feature_reduction qsvm vqc hep higgs",
    url="github.com/qml_hep/ADD_NEW_REPO_NAME",
    license="MIT",
    python_requires=">=3.9",
    zip_safe=False,
    setup_requires="black",
    packages=find_packages(
        include=["autoencoders", "autoencoders.*", "qsvm", "qsvm.*"]
    ),
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Physicists, Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
    ],
)
