import torch
from typing import List, Dict, Optional
import pathlib

from finetrainers.logging import get_logger

logger = get_logger()

# Definition of supported auxiliary data types
# Each entry contains:
# 1. List of valid filenames
# 2. A loader function that processes the file content
auxiliary_datatypes = {
    "noise": {
        "filenames": ["noise.txt", "noises.txt"],
        "loader": torch.load,
    },
    # Other types can be added here in the future
}

def get_auxiliary_data_paths(
    root: pathlib.Path,
    captions: List,
    auxiliary_type: str
) -> List[str]:
    """
    Helper function to get auxiliary data file paths if they exist.

    Args:
        root: Root directory containing the dataset
        captions: List of captions to match auxiliary files with (for length validation)
        auxiliary_type: Type of auxiliary data to load (must be in auxiliary_datatypes)

    Returns:
        List of auxiliary data file paths if available, otherwise empty list
    """
    if auxiliary_type not in auxiliary_datatypes:
        logger.warning(f"Auxiliary data type '{auxiliary_type}' not found in auxiliary_datatypes")
        return []

    valid_filenames = auxiliary_datatypes[auxiliary_type]["filenames"]

    existing_files = [file for file in valid_filenames if (root / file).exists()]

    if not existing_files:
        return []

    if len(existing_files) > 1:
        raise ValueError(
            f"Multiple {auxiliary_type} files found in {root}. Must have exactly one of {valid_filenames}"
        )

    file_list = existing_files[0]

    with open((root / file_list).as_posix(), "r") as f:
        paths = f.read().splitlines()
        paths = [(root / path).as_posix() for path in paths]

    if len(paths) != len(captions):
        raise ValueError(
            f"Number of {auxiliary_type} files ({len(paths)}) must match number of captions ({len(captions)})"
        )

    logger.info(f"Loaded {len(paths)} {auxiliary_type} paths from {file_list}")

    return paths


def load_auxiliary_data(
    auxiliary_type: str,
    path: str,
    device=None,
    dtype=None
) -> torch.Tensor:
    """
    Load and process auxiliary data from a file

    Args:
        auxiliary_type: Type of auxiliary data to load (must be in auxiliary_datatypes)
        path: Path to the auxiliary data file
        device: Optional device to move tensor to
        dtype: Optional dtype to convert tensor to

    Returns:
        torch.Tensor: Processed auxiliary data
    """
    if auxiliary_type not in auxiliary_datatypes:
        raise ValueError(f"Auxiliary data type '{auxiliary_type}' not found in auxiliary_datatypes")

    loader = auxiliary_datatypes[auxiliary_type]["loader"]

    # Load the data
    data = loader(path)

    # Move to device and convert dtype if specified
    if device is not None:
        data = data.to(device=device)

    if dtype is not None:
        data = data.to(dtype=dtype)

    return data
