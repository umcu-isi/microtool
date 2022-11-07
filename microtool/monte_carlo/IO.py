from pathlib import Path
from typing import Tuple


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
