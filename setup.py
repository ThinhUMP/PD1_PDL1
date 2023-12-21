# numpy
# pandas
# seaborn
# matplotlib
# jupyterlab
# jupyter
# tqdm
# rdkit
# rdkit-pypi
# tmap
# pip install git+https://github.com/reymond-group/map4@v1.0
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# scikit-learn
# statannotations
# statannot
# scopy

from setuptools import setup, find_packages

setup(
    name='ann_simi_vanthinh',
    version='2023.12.21',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'seaborn',
        'matplotlib',
        'jupyterlab',
        'jupyter',
        'tqdm',
        'rdkit',
        'rdkit-pypi',
        'tmap',
        'scikit-learn',
        'statannot',
        'scopy',
    ],
    dependency_links=[
        'pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121',
        'pip install git+https://github.com/reymond-group/map4@v1.0',
    ],
    extras_require={
        'gpu': ['torch==1.8.1+cu121', 'torchvision==0.9.1+cu121', 'torchaudio===0.8.1']
    }
)
