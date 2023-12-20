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
        'numpy==1.26.2',
        'pandas==2.1.4',
        'seaborn==0.11.2',
        'matplotlib==3.8.2',
        'jupyterlab==4.0.9',
        'jupyter==1.0.0',
        'tqdm==4.66.1',
        'rdkit==2023.9.2',
        'rdkit-pypi==2022.9.5',
        'tmap==1.2.1',
        'scikit-learn==1.3.2',
        'statannot==0.2.3',
        'scopy==1.2.5',
    ],
    dependency_links=[
        'pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121',
        'pip install git+https://github.com/reymond-group/map4@v1.0',
    ],
    extras_require={
        'gpu': ['torch==1.8.1+cu121', 'torchvision==0.9.1+cu121', 'torchaudio===0.8.1']
    }
)
