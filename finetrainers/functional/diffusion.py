import torch
import rp
from typing import Dict, Optional

from rp.git.CommonSource.noise_warp import resize_noise, mix_new_noise

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
    rp.fansi_print(f"GET_NOISE: latent_model_conditions keys = {list(latent_model_conditions.keys())}", 'cyan bold')
    
    if "noise" in latent_model_conditions:
        # Use custom noise for Go With The Flow
        noise = latent_model_conditions["noise"].to(device=latents.device, dtype=latents.dtype)
        rp.fansi_print(f"GET_NOISE: Using custom noise with shape {noise.shape}, latents shape {latents.shape}", 'green bold')
        
        if noise.shape != latents.shape:
            B, C, T, H, W = latents.shape  # latents are BCTHW
            B_n, T_n, C_n, H_n, W_n = noise.shape  # noise is BTCHW

            rp.fansi_print(f"RESIZING NOISE: old shape = {noise.shape}   --->   new shape == {latents.shape}", 'green orange bold italic on black black')
            assert B==1, 'Only use batch size 1 please, but B=='+str(B)
            
            # Remove batch dimension: BTCHW -> TCHW
            noise_frames = noise[0]  # [T, C, H, W]
            
            # Use built-in 4D batch functionality of resize_noise
            resized_frames = resize_noise(noise_frames, (H, W))  # Handles 4D directly
            
            # Rearrange TCHW -> CTHW and add batch dimension
            import einops
            noise = einops.rearrange(resized_frames, 't c h w -> 1 c t h w')
            
            # Use rp.resize_list to handle temporal dimension change from T_n to T
            noise = rp.resize_list(noise[0], T)[None]

            assert noise.shape==latents.shape

        DEGRADATION_LEVEL = rp.random_float(0,1)
        rp.fansi_print(f"DEGRADATION LEVEL: {DEGRADATION_LEVEL}", 'green orange bold italic on black black')

        noise = mix_new_noise(noise, alpha=DEGRADATION_LEVEL)

    else:
        rp.fansi_print("GET_NOISE: No custom noise found, generating random noise", 'red bold')
        # Generate random noise
        noise = torch.zeros_like(latents).normal_(generator=generator)

    return noise
