from typing import Union

from pathlib2 import Path

from ourlib.image.utils import convert_to_PIL, save_image_to_file

BASE_DIR_PATH = None


def set_base_dir_path(exp_dir_path: Union[str, Path]):
    global BASE_DIR_PATH
    BASE_DIR_PATH = Path(exp_dir_path)


def get_base_dir_path():
    global BASE_DIR_PATH
    assert BASE_DIR_PATH is not None
    return BASE_DIR_PATH


def get_path(subpath: Union[Path, str] = None):
    path = get_base_dir_path()
    if subpath:
        path = path / Path(subpath)
    return path


def save_image_base_dir(image, subpath: Union[Path, str] = None):
    path = get_path(subpath)
    save_image_to_file(image, path)

