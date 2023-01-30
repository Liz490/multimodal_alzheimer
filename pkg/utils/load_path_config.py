import yaml
from pathlib import Path


def load_path_config() -> dict[str, Path]:
    """
    Load paths from yaml file and convert them to Path objects.
    """

    # Load paths from yaml file
    with open('path_config.yaml', 'r') as f:
        paths = yaml.load(f, Loader=yaml.FullLoader)

    # Convert relative paths to absolute paths
    if "relative" in paths:
        for key, value in paths["relative"].items():
            paths[key] = Path.cwd() / value

    # Convert all paths to Path objects
    for key, value in paths.items():
        if key != "relative":
            paths[key] = Path(value)

    return paths
