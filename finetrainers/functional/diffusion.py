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
    rp.fansi_print(f"GET_NOISE: ENTRY - latent_model_conditions keys = {list(latent_model_conditions.keys())}", 'cyan bold')
    rp.fansi_print(f"GET_NOISE: ENTRY - latents shape = {latents.shape}", 'cyan bold')
    
    if "noise" in latent_model_conditions:
        # Use custom noise for Go With The Flow
        original_noise = latent_model_conditions["noise"]
        rp.fansi_print(f"GET_NOISE: FOUND CUSTOM NOISE - original shape = {original_noise.shape}, dtype = {original_noise.dtype}", 'green bold')
        
        noise = original_noise.to(device=latents.device, dtype=latents.dtype)
        rp.fansi_print(f"GET_NOISE: MOVED TO DEVICE - noise shape = {noise.shape}, device = {noise.device}, dtype = {noise.dtype}", 'green bold')
        
        if noise.shape != latents.shape:
            B, C, T, H, W = latents.shape  # latents are BCTHW
            B_n, T_n, C_n, H_n, W_n = noise.shape  # noise is BTCHW

            rp.fansi_print(f"RESIZING NOISE: SHAPE MISMATCH - old shape = {noise.shape} vs target = {latents.shape}", 'yellow bold')
            rp.fansi_print(f"RESIZING NOISE: DIMENSIONS - latents BCTHW=({B},{C},{T},{H},{W}) vs noise BTCHW=({B_n},{T_n},{C_n},{H_n},{W_n})", 'yellow bold')
            assert B==1, 'Only use batch size 1 please, but B=='+str(B)
            
            # Remove batch dimension: BTCHW -> TCHW
            noise_frames = noise[0]  # [T, C, H, W]
            rp.fansi_print(f"RESIZING NOISE: REMOVED BATCH - noise_frames shape = {noise_frames.shape}", 'yellow bold')
            
            # Process each frame individually since 4D batch mode has a bug
            rp.fansi_print(f"RESIZING NOISE: PROCESSING {T_n} FRAMES individually", 'yellow bold')
            resized_frame_list = []
            for i, frame in enumerate(noise_frames):
                rp.fansi_print(f"RESIZING NOISE: FRAME {i+1}/{T_n} - input shape = {frame.shape}", 'yellow bold')
                resized_frame = resize_noise(frame, (H, W))  # frame is CHW
                rp.fansi_print(f"RESIZING NOISE: FRAME {i+1}/{T_n} - output shape = {resized_frame.shape}", 'yellow bold')
                resized_frame_list.append(resized_frame)
            
            resized_frames = torch.stack(resized_frame_list, dim=0)  # Stack back to TCHW
            rp.fansi_print(f"RESIZING NOISE: STACKED ALL FRAMES - shape = {resized_frames.shape}", 'yellow bold')
            
            # Rearrange TCHW -> CTHW and add batch dimension
            import einops
            noise = einops.rearrange(resized_frames, 't c h w -> 1 c t h w')
            rp.fansi_print(f"RESIZING NOISE: AFTER REARRANGE - shape = {noise.shape}", 'yellow bold')
            
            # Use rp.resize_list to handle temporal dimension change from T_n to T
            rp.fansi_print(f"RESIZING NOISE: TEMPORAL RESIZE from {T_n} to {T} frames", 'yellow bold')
            noise = rp.resize_list(noise[0], T)[None]
            rp.fansi_print(f"RESIZING NOISE: AFTER TEMPORAL RESIZE - final shape = {noise.shape}", 'yellow bold')

            assert noise.shape==latents.shape, f"Shape mismatch after resize: {noise.shape} vs {latents.shape}"
            rp.fansi_print(f"RESIZING NOISE: SHAPE MATCH CONFIRMED ✓", 'green bold')

        DEGRADATION_LEVEL = rp.random_float(0,1)
        rp.fansi_print(f"DEGRADATION LEVEL: {DEGRADATION_LEVEL}", 'green orange bold italic on black black')

        rp.fansi_print(f"GET_NOISE: BEFORE mix_new_noise - shape = {noise.shape}", 'magenta bold')
        noise = mix_new_noise(noise, alpha=DEGRADATION_LEVEL)
        rp.fansi_print(f"GET_NOISE: AFTER mix_new_noise - shape = {noise.shape}", 'magenta bold')

        rp.fansi_print(f"GET_NOISE: RETURNING CUSTOM NOISE - final shape = {noise.shape}, min = {noise.min():.4f}, max = {noise.max():.4f}, mean = {noise.mean():.4f}", 'green bold')

    else:
        rp.fansi_print("🚨🚨🚨 GET_NOISE: FALLBACK TO RANDOM NOISE - THIS IS BAD! 🚨🚨🚨", 'red bold on white')
        rp.fansi_print("🚨🚨🚨 CUSTOM NOISE SYSTEM FAILED - CHECK DATA PIPELINE! 🚨🚨🚨", 'red bold on white')
        # Generate random noise
        noise = torch.zeros_like(latents).normal_(generator=generator)
        rp.fansi_print(f"🚨🚨🚨 FALLBACK RANDOM NOISE - shape = {noise.shape}, min = {noise.min():.4f}, max = {noise.max():.4f}, mean = {noise.mean():.4f} 🚨🚨🚨", 'red bold on white')

    return noise
