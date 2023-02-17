from os import path

import setuptools

path_to_repo = path.abspath(path.dirname(__file__))
with open(path.join(path_to_repo, 'readme.md'), encoding='utf-8') as f:
    long_description = f.read()

required_pypi = [
    'numpy',
    'scikit-learn',
    'pandas',
    'tqdm',
    'dict_hash', # required for caching
    'transformers',
    'torch',
    'imodelsx',
    'langchain',
    'openai',
    'accelerate',

    'datasets', # optional, required for getting NLP datasets
    'pytest', # optional, required for running tests
]

setuptools.setup(
    name="mprompt",
    version="0.01",
    author="Microsoft Research",
    author_email="",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/csinva/module-prompt",
    packages=setuptools.find_packages(
        exclude=['tests', 'tests.*', '*.test.*']
    ),
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=required_pypi,
)
