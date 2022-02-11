import setuptools

setuptools.setup(
    name="MICROtool",
    version="0.0.1",
    author="Edwin Bennink",
    author_email="H.E.Bennink@umcutrecht.nl",
    description="MICROtool Framework for Diffusion MRI Experiment Optimization",
    packages=['microtool'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: Microsoft :: Windows",
        "Development Status :: 2 - Pre-Alpha"
    ],
    python_requires='>=3.8',
    install_requires=['numpy'],
    extras_require={
        'dmipy support': ['dmipy']
    }
)
