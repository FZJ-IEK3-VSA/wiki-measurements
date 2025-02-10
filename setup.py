import os
from setuptools import setup, find_packages

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, "README.md"), "r") as fh:
    long_description = fh.read()

setup(
    name="wikimeasurements",
    version="0.0.0",
    license="",
    author="Jan Göpfert",
    author_email="j.goepfert@fz-juelich.de",
    description="Create NLP datasets for quantity span identification and measurement context extraction from Wikipedia and Wikidata",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    include_package_data=True,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    # setup_requires=["setuptools-git"], # what is this for?
    python_requires=">=3.8",
    classifiers=[
        # check http://pypi.python.org/pypi?%3Aaction=list_classifiers
        # License :: OSI Approved :: MIT License
        "Development Status :: 2 - Pre-Alpha"
        "Programming Language :: Python :: 3 :: Only"
        "Operating System :: Unix"
        "Natural Language :: English"
        "Intended Audience :: Science/Research"
        "Intended Audience :: Information Technology"
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords=[
        "NLP",
        "Distant Supervision",
        "Wikipedia",
        "Wikidata",
        "Quantities",
        "Measurements",
        "Information Extraction",
    ],
)
