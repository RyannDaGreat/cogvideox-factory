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

            rp.fansi_print(f"RESIZING NOISE: {noise.shape} -> {latents.shape} | Spatial: ({H_n}x{W_n})->({H}x{W}) | Temporal: {T_n}->{T}", 'yellow bold')
            assert B==1, 'Only use batch size 1 please, but B=='+str(B)
            
            # Remove batch dimension: BTCHW -> TCHW
            noise_frames = noise[0]  # [T, C, H, W]
            
            # Check if spatial resizing is needed
            spatial_resize_needed = (H_n != H) or (W_n != W)
            
            if spatial_resize_needed:
                rp.fansi_print(f"RESIZING NOISE: Spatial resize needed, processing {T_n} frames...", 'yellow bold')
                resized_frame_list = []
                for i, frame in enumerate(noise_frames):
                    if i == 0:  # Only log first frame
                        rp.fansi_print(f"RESIZING NOISE: Frame 1 - {frame.shape} on {frame.device}", 'yellow bold')
                    # Move frame to CPU for resize_noise (it uses CPU coordinate matrices)
                    frame_cpu = frame.cpu()
                    resized_frame = resize_noise(frame_cpu, (H, W))  # frame is CHW
                    # Move back to original device
                    resized_frame = resized_frame.to(frame.device)
                    if i == 0:  # Only log first frame result
                        rp.fansi_print(f"RESIZING NOISE: Frame 1 result - {resized_frame.shape} on {resized_frame.device}", 'yellow bold')
                    resized_frame_list.append(resized_frame)
                
                resized_frames = torch.stack(resized_frame_list, dim=0)  # Stack back to TCHW
            else:
                rp.fansi_print(f"RESIZING NOISE: No spatial resize needed, skipping", 'yellow bold')
                resized_frames = noise_frames
            
            # Rearrange TCHW -> CTHW and add batch dimension
            import einops
            noise = einops.rearrange(resized_frames, 't c h w -> 1 c t h w')
            
            # Use rp.resize_list to handle temporal dimension change from T_n to T
            if T_n != T:
                rp.fansi_print(f"RESIZING NOISE: Temporal resize {T_n} -> {T} frames", 'yellow bold')
                # Remove batch dim: [1, C, T, H, W] -> [C, T, H, W]
                noise_no_batch = noise[0]
                rp.fansi_print(f"RESIZING NOISE: Before temporal resize - shape = {noise_no_batch.shape}", 'yellow bold')
                
                # rp.resize_list operates on the first dimension, so we need to rearrange
                # [C, T, H, W] -> [T, C, H, W] -> resize -> [T_new, C, H, W] -> [C, T_new, H, W]
                noise_transposed = noise_no_batch.permute(1, 0, 2, 3)  # [T, C, H, W]
                rp.fansi_print(f"RESIZING NOISE: Transposed for resize - shape = {noise_transposed.shape}", 'yellow bold')
                
                noise_resized_t = rp.resize_list(noise_transposed, T)  # Resize temporal dimension
                rp.fansi_print(f"RESIZING NOISE: After resize_list - shape = {noise_resized_t.shape}", 'yellow bold')
                
                noise_resized = noise_resized_t.permute(1, 0, 2, 3)  # [C, T_new, H, W]
                rp.fansi_print(f"RESIZING NOISE: Transposed back - shape = {noise_resized.shape}", 'yellow bold')
                
                noise = noise_resized[None]  # Add batch back: [1, C, T, H, W]
            else:
                rp.fansi_print(f"RESIZING NOISE: No temporal resize needed", 'yellow bold')

            assert noise.shape==latents.shape, f"Shape mismatch after resize: {noise.shape} vs {latents.shape}"
            rp.fansi_print(f"RESIZING NOISE: ✓ Final shape = {noise.shape}", 'green bold')

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
