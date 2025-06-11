import torch
import rp
from typing import Dict, Optional


def flow_match_xt(x0: torch.Tensor, n: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    r"""Forward process of flow matching."""
    return (1.0 - t) * x0 + t * n


def flow_match_target(n: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    r"""Loss target for flow matching."""
    return n - x0


def get_noise(
    latents: torch.Tensor,
    latent_model_conditions: Dict[str, torch.Tensor],
    generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    r"""
    Get noise tensor for diffusion process, prioritizing custom noise if available.

    Args:
        latents: The latent tensor to match noise shape with
        latent_model_conditions: Dictionary containing model conditions, which may include "noise"
        generator: Optional random generator for reproducibility

    Returns:
        torch.Tensor: The noise tensor (either custom or randomly generated),
    """
    if "noise" in latent_model_conditions:
        # Use custom noise for Go With The Flow
        noise = latent_model_conditions["noise"].to(device=latents.device, dtype=latents.dtype)
        if noise.shape != latents.shape:
            from rp.git.CommonSource.noise_warp import resize_noise
            B, T, C, H, W = latents.shape

            rp.fansi_print(f"RESIZING NOISE: old shape = {noise.shape}   --->   new shape == {latents.shape}", 'green orange bold italic on black black')
            assert B==1, 'Only use batch size 1 please, but B=='+str(B)
            noise = resize_noise(noise, (H, W))
            noise = rp.resize_list(noise[0], T)[None]

            assert noise.shape==latents.shape

            raise ValueError(f"Custom noise shape {noise.shape} does not match latent shape {latents.shape}")
    else:
        # Generate random noise
        noise = torch.zeros_like(latents).normal_(generator=generator)

    return noise
