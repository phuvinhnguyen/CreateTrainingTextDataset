# setup.py

from setuptools import setup, find_packages

setup(
    name='processTrainingText',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # List any dependencies here
        'transformers',
        'datasets',
        'tqdm',
    ],
)
