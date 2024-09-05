import warnings
from pathlib import Path
from typing import Tuple, Union


def make_result_dirs(result_directory: Path, category_name: str) -> Tuple[Path, Path]:
    """
    Makes a subfolder in the result directory with catergory_name, and subsequently makes subfolders in that folder
    with names plots and data

    :param result_directory: Path to result directory
    :param category_name: name for the subdirectory to be created
    :return: plotsubdir, datasubdir
    """
    result_directory.mkdir(exist_ok=True)
    outputdir = result_directory / category_name
    outputdir.mkdir(exist_ok=True)
    plotdir = outputdir / "plots"
    datadir = outputdir / "data"
    plotdir.mkdir(exist_ok=True)
    datadir.mkdir(exist_ok=True)
    return plotdir, datadir


def make_expirement_directories(parent_directory: Union[Path, str], expirement_name: str):
    """

    :param parent_directory:
    :param expirement_name:
    :return: plotdir,modeldir,simdir,schemedir
    """
    if isinstance(parent_directory, str):
        parent_directory = Path(parent_directory)

    exp_dir = parent_directory / expirement_name
    all_sub_dirs = get_experiment_subdirs(exp_dir)
    if exp_dir.is_dir():
        warnings.warn("There is already a folder with the name of the experiment. No new folders have been created.")
    else:
        exp_dir.mkdir()
        for sub_dir in all_sub_dirs:
            sub_dir.mkdir()

    return all_sub_dirs


def get_experiment_subdirs(exp_dir):
    """
    For extracting the subdirectories associated with an experiment directory

    :param exp_dir: Path to an experiment directory
    :return: paths to plots,models,schemes, simulations directories
    """
    if isinstance(exp_dir, str):
        exp_dir = Path(exp_dir)

    plotdir = exp_dir / "plots"
    modeldir = exp_dir / "models"
    schemedir = exp_dir / "schemes"
    simdir = exp_dir / "simulations"
    return plotdir, modeldir, simdir, schemedir
