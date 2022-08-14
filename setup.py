from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
from setuptools import find_packages, setup
import platform
import pathlib


class MyBdistWheel(_bdist_wheel):
    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        self.root_is_pure = False


if __name__ == "__main__":
    os_tag = {
        "Windows": "win_amd64",
        "Darwin": "macosx_x86_64",
        "Linux": "manylinux1_x86_64"
    }

    # The directory containing this file
    # Reads the content of your README.md into a variable to be used in the setup below
    HERE = pathlib.Path(__file__).parent

    # The text of the README file
    README = (HERE / "README.md").read_text()

    install_requirement = [
        "gensim >= 4.1.2",
        "nltk >= 3.5",
        "scikit-learn >= 0.23.2",
        "stanfordcorenlp >= 3.9.1.1",
        "tabulate >= 0.8.9",
        "matplotlib >= 3.1.1",
        "torch >= 1.11.0",
    ]

    # This call to setup() does all the work
    setup(
        author="Vitor Meriat",
        author_email="vitormeriat@gmail.com",
        # Should match the package folder
        name="text4gcn",
        # Should match the package folder
        # Packages=["text4gcn"],
        packages=find_packages(
            ".",
            exclude=(
                "examples.*",
                "examples",
                "tests",
                "test.py",
            ),
        ),
        # Important for updates
        version="1.0.0",
        # Should match your chosen license
        license="MIT",
        description="A library for building text graphs for the application of Graph Neural Networks (GNN), in the context of text classification in natural language processing",
        # Loads your README.md
        long_description=README,
        # README.md is of type 'markdown'
        long_description_content_type="text/markdown",
        url="https://github.com/vitormeriat/text4gcn",
        keywords=["tgcn", "gcn", "gnn", "nlp",
                  "text_classification", "text_graph"],
        classifiers=[
            "License :: OSI Approved :: MIT License",
            #"Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Operating System :: OS Independent",
        ],
        include_package_data=True,
        install_requires=install_requirement,
        platforms=os_tag[platform.system()],
        # entry_points={
        #     "console_scripts": [
        #         "text4gcn=text4gcn.__main__:main",
        #     ]
        # },
    )
    print(
        "Text4GCN Python library installation finished. Please manually check Stanford CoreNLP"
        "(https://stanfordnlp.github.io/CoreNLP/) is installed and "
        "running in your environment."
    )
