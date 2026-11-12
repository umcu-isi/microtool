import setuptools

setuptools.setup(
    name="MICROtool",
    version="0.1.0",
    author="Edwin Bennink, Frank W.H. Otto",
    author_email="H.E.Bennink@umcutrecht.nl",
    description="MICROtool Framework for Diffusion MRI Experiment Optimization",
    packages=['microtool'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
    python_requires='>=3.11',
    install_requires=[
        'numpy>=1.24,<2.0',
        'scipy>=1.9,<1.14',
        'tabulate~=0.9',
        'numba>=0.58,<0.60',
        'tqdm~=4.66',
        'pandas>=2.0,<2.2.3',
        'matplotlib>=3.7,<3.9',
    ],
    extras_require={
        'dmipy': ['dmipy~=3.0'],
        'microtorch': ['microtorch-mri~=0.1.1'],
        'testing': ['pint~=0.21'],
    },
)