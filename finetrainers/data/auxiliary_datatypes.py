import torch
from typing import List, Dict, Optional, Union, Any
import pathlib

from finetrainers.logging import get_logger

import einops

logger = get_logger()

import rp
import numpy as np

# Definition of supported auxiliary data types
# Each entry contains:
# 1. List of valid filenames
# 2. A loader function that processes the file content

def load_noise(file):
    if file.endswith('.pth'):
        return torch.load(file)
    elif file.endswith('.npy'):
        print("LOADING NOISE FROM",file)
        output = np.load(file)
        assert output.ndim==4, 'THWC'
        output = torch.from_numpy(output)
        output = einops.rearrange(output, 'T H W C -> T C H W')
        return output

AUXILIARY_DATATYPES = {
    "noise": {
        "filenames": ["noise.txt", "noises.txt"],
        "loader": load_noise,
    },
    # Other types can be added here in the future
}

def load_auxiliary_data_paths(
    dataset_root: pathlib.Path,
    captions: List,
    auxiliary_data_types: Optional[List[str]] = None
) -> Dict[str, List[str]]:
    """
    Helper function to load all auxiliary data paths for multiple types.

    Args:
        dataset_root: Root directory containing the dataset
        captions: List of captions to match auxiliary files with (for length validation)
        auxiliary_data_types: List of auxiliary data types to load (defaults to empty list)

    Returns:
        Dictionary mapping auxiliary types to their sample file paths (the .mp4 or .pth files etc)
    """
    auxiliary_data_paths = {}

    # Default to empty list if None is provided
    if auxiliary_data_types is None:
        return auxiliary_data_paths

    for aux_type in auxiliary_data_types:
        paths = get_auxiliary_data_paths(dataset_root, captions, aux_type)
        if paths:
            auxiliary_data_paths[aux_type] = paths
            logger.info(f"Loaded auxiliary data type: {aux_type}")

    return auxiliary_data_paths


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
        List of auxiliary data file paths (i.e. text files containing one line for each sample)

    Raises:
        ValueError if no file lists can be found, there are multiple file lists
        (such as finding both "noises.txt" and "noise.txt" in the data root at the same time),
        or the number of files in that list mismatches the number of captions in the dataset
    """
    if auxiliary_type not in AUXILIARY_DATATYPES:
        raise ValueError(
            f"Auxiliary data type '{auxiliary_type}' not found in auxiliary_datatypes"
        )

    valid_filenames = AUXILIARY_DATATYPES[auxiliary_type]["filenames"]

    existing_files = [file for file in valid_filenames if (root / file).exists()]

    if len(existing_files) == 0:
        raise ValueError(
            f"No {auxiliary_type} files found in {root}. Must have exactly one of {valid_filenames}"
        )
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


def process_auxiliary_data_for_sample(
    sample: dict,
    auxiliary_data_paths: Dict[str, List[str]],
    sample_index: int,
) -> dict:
    """
    Process all available auxiliary data for a sample at a given index

    Args:
        sample: The sample dictionary to update with auxiliary data
        auxiliary_data_paths: Dictionary mapping auxiliary types to their file paths
        sample_index: Current sample index in the dataset

    Returns:
        Updated sample with auxiliary data added
    """
    for aux_type, paths in auxiliary_data_paths.items():
        path = paths[sample_index]
        sample[aux_type] = load_auxiliary_data(aux_type, path)

    return sample


def load_auxiliary_data(
    auxiliary_type: str,
    path: str,
) -> Union[Any, torch.Tensor]:
    """
    Load and process auxiliary data from a file

    Args:
        auxiliary_type: Type of auxiliary data to load (must be in auxiliary_datatypes)
        path: Path to the auxiliary data file

    Returns:
        Usually a torch.Tensor: Processed auxiliary data; return type depends on the datatype's "loader" function
    """
    if auxiliary_type not in AUXILIARY_DATATYPES:
        raise ValueError(f"Auxiliary data type '{auxiliary_type}' not found in auxiliary_datatypes")

    loader = AUXILIARY_DATATYPES[auxiliary_type]["loader"]

    # Load the data
    data = loader(path)

    return data
