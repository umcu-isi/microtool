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
        "Operating System :: Microsoft :: Windows",
        "Development Status :: 3 - Alpha"
    ],
    python_requires='==3.8.*',
    install_requires=[
        'numpy~=1.24',
        'scipy~=1.9',
        'tabulate~=0.9',
        'numba~=0.58',
        'tqdm~=4.66',
        'pandas~=2.0',
        'matplotlib~=3.7',
    ],
    extras_require={
        'dmipy support': ['dmipy~=1.0'],
        'testing': ['pint~=0.21'],
    }
)
