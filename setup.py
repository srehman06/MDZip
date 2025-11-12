# setup.py (add entry_points)
from setuptools import setup, find_packages

setup(
    name='mdzip',
    version='0.1.1',
    author='Namindu De Silva',
    author_email='nami.rangana@gmail.com',
    description='Compress MD trajectories using deep convolutional autoencoder',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/nami-rangana/MolZip.git',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'wheel','mdtraj','mdanalysis','torch','torchvision','torchaudio',
        'pytorch-lightning','scikit-learn','numpy','tqdm', 'tensorboardX',
        'tensorboard'
    ],
    entry_points={
        "console_scripts": [
            "mdzip=mdzip.mdzip_core:main",
        ]
    },
)