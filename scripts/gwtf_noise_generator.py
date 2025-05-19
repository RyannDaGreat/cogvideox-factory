#!/usr/bin/env python3
# Generate custom noise tensors for Go With The Flow training

import torch
import os
from pathlib import Path
from tqdm import tqdm
import argparse

def generate_gwtf_noise_tensors(
    dataset_path: str, 
    output_path: str, 
    shape,
    seed: int = 42,
    overwrite: bool = False
):
    """
    Generate noise tensors for Go With The Flow training.
    
    Args:
        dataset_path: Path to the dataset directory
        output_path: Path to save the noise tensors
        shape: Shape of the noise tensors to generate (e.g., [4, 16, 64, 64])
        seed: Random seed for reproducibility
        overwrite: Whether to overwrite existing noise tensors
    """
    os.makedirs(output_path, exist_ok=True)
    output_path = Path(output_path)
    
    # Get list of files in dataset
    dataset_path = Path(dataset_path)
    files = list(dataset_path.glob('**/*.mp4')) + list(dataset_path.glob('**/*.jpg'))
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    for file in tqdm(files, desc="Generating GWTF noise tensors"):
        # Generate a consistent sample ID from the file path
        sample_id = file.stem
        output_file = output_path / f"{sample_id}.pt"
        
        # Skip if the file already exists and we're not overwriting
        if output_file.exists() and not overwrite:
            continue
            
        # Generate a noise tensor with the specified shape
        noise = torch.randn(shape)
        
        # Save the noise tensor
        torch.save(noise, output_file)
        
    print(f"Generated {len(files)} noise tensors for GWTF at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate noise tensors for Go With The Flow training")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the noise tensors")
    parser.add_argument("--shape", type=int, nargs="+", required=True, help="Shape of the noise tensors to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing noise tensors")
    
    args = parser.parse_args()
    
    generate_gwtf_noise_tensors(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        shape=args.shape,
        seed=args.seed,
        overwrite=args.overwrite
    )