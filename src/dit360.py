import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional

from render import (
    hunyuanworld_to_pointcloud,
    render_equirectangular_torch,
    pers_to_equi,
    c2w_to_w2c,
)
from mask_utils import clean_mask, fill_holes_with_interpolation


def prepare_dit360_inputs(
    predictions: Dict,
    frame_idx: int = 0,
    H: int = 1024,
    W: int = 2048,
    num_views: int = 20,
    confidence_threshold: float = 0.1,
    subsample_ratio: float = 0.5,
    kernel_size: int = 3,
    min_region_size: int = 100,
) -> Tuple[Image.Image, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare DiT360 inputs from HunyuanWorld predictions
    
    Args:
        predictions: HunyuanWorld predictions dict with input_images
        frame_idx: Index of reference frame
        H, W: Equirectangular dimensions
        num_views: Number of views for point cloud
        confidence_threshold: Point confidence threshold
        subsample_ratio: Point subsampling ratio
        kernel_size: Morphological kernel size
        min_region_size: Minimum connected component size
        
    Returns:
        init_image: PIL Image for DiT360
        mask_np: [H, W] Mask for DiT360 (uint8, 255=keep, 0=outpaint)
        rendered_mask: [H, W] Boolean mask where point cloud rendered
        rendered_depth: [H, W] Depth map from point cloud rendering
    """
    input_images = predictions['input_images']
    camera_poses = predictions["camera_poses"][0].cpu().numpy()
    camera_intrs = predictions["camera_intrs"][0].cpu().numpy()
    
    points, colors = hunyuanworld_to_pointcloud(
        predictions, input_images, num_views=num_views,
        confidence_threshold=confidence_threshold, subsample_ratio=subsample_ratio
    )
    
    cam_c2w = camera_poses[frame_idx]
    w2c_pose = c2w_to_w2c(cam_c2w)
    
    rendered_rgb, rendered_depth, rendered_mask = render_equirectangular_torch(
        points, colors, H=H, W=W, camera_pose=w2c_pose
    )
    
    first_rgb_frame = input_images[0, frame_idx].permute(1, 2, 0).cpu().numpy()
    if first_rgb_frame.max() <= 1.0:
        first_rgb_frame = (first_rgb_frame * 255).astype(np.uint8)
    
    intrinsics = camera_intrs[frame_idx] if camera_intrs.ndim == 3 else camera_intrs
    equi_projected_pers_input, projected_mask = pers_to_equi(first_rgb_frame, intrinsics, H, W)
    
    combined_rgb = rendered_rgb.copy()
    valid_projected = projected_mask & (equi_projected_pers_input.sum(axis=2) > 0)
    combined_rgb[valid_projected] = equi_projected_pers_input[valid_projected]
    
    refined_mask = clean_mask(rendered_mask, kernel_size=kernel_size, min_region_size=min_region_size)
    filled_rgb = fill_holes_with_interpolation(combined_rgb, rendered_mask, refined_mask)
    
    mask_np = (refined_mask * 255).astype(np.uint8)
    init_image = Image.fromarray(filled_rgb.astype(np.uint8))
    
    return init_image, mask_np, rendered_mask, rendered_depth


class DiT360:
    def __init__(
        self,
        model_name: str = "black-forest-labs/FLUX.1-dev",
        lora_name: str = "Insta360-Research/DiT360-Panorama-Image-Generation",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self.device)
        self.dtype = torch.float16
        self.cache_dir = cache_dir
        
        print(f"loading DiT360 model {model_name}...")
        from image_outpaint.DiT360.pa_src.pipeline import RFPanoInversionParallelFluxPipeline
        
        self.pipe = RFPanoInversionParallelFluxPipeline.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir
        ).to(self.device)
        self.pipe.load_lora_weights(lora_name)

        print("DiT360 model loaded")
    
    def __call__(
        self,
        init_image: Image.Image,
        mask_np: np.ndarray,
        prompt: str,
        height: int = 1024,
        width: int = 2048,
        timestep: int = 50,
        seed: int = 0,
        tau: int = 50,
    ) -> Image.Image:
        """        
        Args:
            init_image: PIL Image for DiT360
            mask_np: [H, W] Mask for DiT360 (uint8, 255=keep, 0=outpaint)
            prompt: Text prompt for outpainting
            height: Image height
            width: Image width
            timestep: Number of diffusion steps
            seed: Random seed
            tau: Range 0-100, smaller = stronger image consistency but may reduce quality
            
        Returns:
            outpainted_image: PIL Image with outpainted result
        """

        from image_outpaint.DiT360.pa_src.attn_processor import PersonalizeAnythingAttnProcessor, set_flux_transformer_attn_processor
        
        latent_h = height // (self.pipe.vae_scale_factor * 2)
        latent_w = width // (self.pipe.vae_scale_factor * 2)
        img_dims = latent_h * (latent_w + 2)
        
        mask_full = torch.tensor(np.where(mask_np == 255, 1, 0), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        block_h = height // latent_h
        block_w = width // latent_w
        
        kernel = torch.ones(1, 1, block_h, block_w, device=mask_full.device)
        conv_result = F.conv2d(mask_full, kernel, stride=(block_h, block_w))
        mask_block = (conv_result == (block_h * block_w)).float()
        
        mask = mask_block[0, 0]
        mask = torch.cat([mask[:, 0:1], mask, mask[:, -1:]], dim=-1).view(-1, 1)
        
        inverted_latents, image_latents, latent_image_ids = self.pipe.invert(
            source_prompt="",
            image=init_image,
            height=height,
            width=width,
            num_inversion_steps=timestep,
            gamma=1.0
        )
        
        set_flux_transformer_attn_processor(
            self.pipe.transformer,
            set_attn_proc_func=lambda name, dh, nh, ap: PersonalizeAnythingAttnProcessor(
                name=name, tau=tau/100, mask=mask, device=self.device, img_dims=img_dims
            ),
        )
        
        image = self.pipe(
            [prompt, prompt],
            inverted_latents=inverted_latents,
            image_latents=image_latents,
            latent_image_ids=latent_image_ids,
            height=height,
            width=width,
            start_timestep=0.0,
            stop_timestep=0.99,
            num_inference_steps=timestep,
            eta=1.0,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            mask=mask,
            use_timestep=True
        ).images[1]
        
        return image
