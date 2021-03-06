from setuptools import find_packages, setup
import pathlib


# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="text4gcn",
    # packages=["text4gcn"],
    packages=find_packages("."),
    version="1.0.1",
    keywords=["pypi", "mikes_toolbox", "tutorial"],
    description="Read the latest Real Python tutorials",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/vitormeriat/text4gcn",
    author="Vitor Meriat",
    author_email="vitormeriat@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        #"Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    include_package_data=True,
    #package_data={'': ['data/*.txt', 'data/*.meta']},
    install_requires=[
        "scikit-learn==1.0.2", 
        "nltk==3.7", 
        "stanfordcorenlp==3.9.1.1", 
        "gensim==4.1.2", 
        "tabulate==0.8.9", 
        "torch==1.11.0", 
        "matplotlib==3.5.1"
    ],
    entry_points={
        "console_scripts": [
            "text4gcn=text4gcn.__main__:main",
        ]
    }
)
print(
    "Text4GCN Python library installation finished. Please manually check Stanford CoreNLP"
    "(https://stanfordnlp.github.io/CoreNLP/) is installed and "
    "running in your environment."
)
