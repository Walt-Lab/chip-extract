"""
Code for loading/interacting with the configuration YAML files.
"""

from pathlib import Path

import yaml


def get_experiment_path(script_path: str) -> Path:
    """
    Load the base path for the current experiment

    Parameters
    ----------
    script_path : str
        Path of script calling this function

    Returns
    -------
    out : pathlib.Path
        Path to folder with experimental images
    """
    config_path = Path(script_path).parents[1] / "config.yml"

    if not config_path.exists():
        raise FileNotFoundError("Check that config.yml exists in the right location.")

    with open(config_path) as f:
        config = yaml.full_load(f)

    return Path(config["base_path"]).resolve()


def load_experiment_config(base_path: Path) -> dict:
    """
    Load a yaml config file

    Parameters
    ----------
    base_path : str
        Path to folder containing experimental images

    Returns
    -------
    out : dict
        Dictionary of parameter values
    """
    config_path = base_path / "experiment_config.yml"

    if not config_path.exists():
        raise FileNotFoundError(
            "Ensure the file 'experiment_config.yml' exists in the experimental images folder!"
            f"\nProvided path:\n{config_path}"
        )

    with open(config_path, "r") as f:
        config = yaml.full_load(f)

    return config


def extract_params(config: dict, param: str) -> list:
    """
    Extract a list of parameters from config file

    Parameters
    ----------
    config : dict
        Dictionary of parameter values loaded from `get_config`
    param : str
        String value of config parameter to extract

    Returns
    -------
    out : list
        List of parameter values
    """
    if all(val is None for val in config[param].values()):
        return None
    else:
        params = []
        if config[param]["negative"] is not None:
            params.append(config[param]["negative"])
        for label in config[param]["experimental"]:
            params.append(label)

        return params
