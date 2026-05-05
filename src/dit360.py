import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional
import os

from render import (
    hunyuanworld_to_pointcloud,
    render_equirectangular_torch,
    c2w_to_w2c,
    render_3dgs_alpha_with_radius_filter,
    render_3dgs_to_equirectangular,
)
from update_3dgs import convert_hunyuanworld_to_gsplat
from mask_utils import build_mask_from_alpha, fill_holes_with_alpha_blending


def find_least_covered_frame_idx(
    predictions: Dict,
    interpolated_camera_poses: Optional[np.ndarray] = None,
    H: int = 512,
    W: int = 1024,
    face_resolution: int = 256,
    alpha_threshold: float = 0.5,
) -> int:
    sparse_poses = predictions["camera_poses"][0].cpu().numpy()
    gsplat_splats = convert_hunyuanworld_to_gsplat(predictions["splats"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    coverage_counts = []
    for cam_c2w in sparse_poses:
        gs_alpha = render_3dgs_alpha_with_radius_filter(
            splats=gsplat_splats,
            w2c_pose=c2w_to_w2c(cam_c2w),
            equi_height=H,
            equi_width=W,
            face_resolution=face_resolution,
            device=device,
            max_gaussian_radius_px=max(48.0, face_resolution * 0.12),
            radius_percentile=99.0,
            scale_percentile=99.5,
        )
        coverage_counts.append(int((gs_alpha > alpha_threshold).sum()))

    best_sparse_idx = int(np.argmin(coverage_counts))

    if interpolated_camera_poses is None:
        print(f"Using frame_idx={best_sparse_idx}")
        return best_sparse_idx

    N_sparse = len(sparse_poses)
    N_interp = len(interpolated_camera_poses)
    best_interp_idx = int(round(best_sparse_idx * (N_interp - 1) / max(N_sparse - 1, 1)))
    print(f"Using frame_idx={best_interp_idx}")
    return best_interp_idx

def prepare_dit360_inputs(
    predictions: Dict,
    frame_idx: int = 0,
    interpolated_camera_poses: Optional[np.ndarray] = None,
    H: int = 1024,
    W: int = 2048,
    num_views: int = 20,
    confidence_threshold: float = 0.1,
    subsample_ratio: float = 1.0,
    kernel_size: int = 6,
    min_region_size: int = 100,
    face_resolution: int = 512,
) -> Tuple[Image.Image, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare DiT360 inputs using only 3DGS alpha for the final keep-mask.
    
    Args:
        predictions: HunyuanWorld predictions dict with input_images and splats
        frame_idx: Index of reference frame
        H, W: Equirectangular dimensions
        num_views: Number of views for point cloud depth generation
        confidence_threshold: Point confidence threshold
        subsample_ratio: Point subsampling ratio
        kernel_size: Morphological kernel size used in alpha mask cleanup
        min_region_size: Minimum connected component size for final keep-mask
        face_resolution: Cubemap face resolution for 3DGS rendering
        
    Returns:
        init_image: PIL Image for DiT360
        mask_np: [H, W] Mask for DiT360 (uint8, 255=keep, 0=outpaint)
        refined_render_mask: [H, W] Boolean mask derived from filtered 3DGS alpha
        rendered_depth: [H, W] Depth map from point cloud rendering
    """
    
    input_images = predictions['input_images']
    if interpolated_camera_poses is not None:
        camera_poses = interpolated_camera_poses
    else:
        camera_poses = predictions["camera_poses"][0].cpu().numpy()
    
    # Step 1: Render point cloud for depth alignment only.
    points, colors = hunyuanworld_to_pointcloud(
        predictions, input_images, num_views=num_views,
        confidence_threshold=confidence_threshold, subsample_ratio=subsample_ratio
    )
    
    cam_c2w = camera_poses[frame_idx]
    w2c_pose = c2w_to_w2c(cam_c2w)
    
    # Render point cloud only to get depth.
    _, rendered_depth, _ = render_equirectangular_torch(
        points, colors, H=H, W=W, camera_pose=w2c_pose
    )

    # Step 2: Convert HunyuanWorld splats to gsplat format.
    gsplat_splats = convert_hunyuanworld_to_gsplat(predictions['splats'])

    # Step 3: Render the full 3DGS for RGB and a filtered 3DGS for mask alpha.
    rendered_3dgs, _ = render_3dgs_to_equirectangular(
        splats=gsplat_splats,
        w2c_pose=w2c_pose,
        equi_height=H,
        equi_width=W,
        face_resolution=face_resolution,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        alpha_threshold=None,
    )

    gs_alpha = render_3dgs_alpha_with_radius_filter(
        splats=gsplat_splats,
        w2c_pose=w2c_pose,
        equi_height=H,
        equi_width=W,
        face_resolution=face_resolution,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        max_gaussian_radius_px=max(48.0, face_resolution * 0.20),
        radius_percentile=99.5,
        scale_percentile=99.5,
    )

    # Step 4: Build the final keep-mask from filtered 3DGS alpha only.
    refined_render_mask = build_mask_from_alpha(
        gs_alpha,
        low_threshold=0.5,
        high_threshold=0.9,
        kernel_size=kernel_size,
        min_region_size=max(min_region_size, 256),
        min_core_pixels=64,
        min_core_ratio=0.01,
        max_diffuse_area_ratio=0.35,
    )
    mask_np = (refined_render_mask * 255).astype(np.uint8)

    # Step 5: Fill weak-alpha gaps inside the keep-mask
    filled_rgb = fill_holes_with_alpha_blending(
        rgb=rendered_3dgs,
        alpha=gs_alpha,
        target_mask=refined_render_mask,
        alpha_threshold=0.8,
    ).astype(np.uint8)
    filled_rgb[~refined_render_mask] = 0
    init_image = Image.fromarray(filled_rgb.astype(np.uint8))

    output_dir = "/projectnb/ivc-ml/mwakeham/panorama/assets/argus_eval/our_results_real/0002_pers/output_intermediates_v2"
    os.makedirs(output_dir, exist_ok=True)
    Image.fromarray((rendered_depth > 0).astype(np.uint8) * 255).save(output_dir + "/1_pointcloud_depth_support.png")
    Image.fromarray(rendered_3dgs).save(output_dir + "/3_3dgs_render.png")
    Image.fromarray((np.clip(gs_alpha, 0.0, 1.0) * 255).astype(np.uint8)).save(output_dir + "/4_3dgs_alpha.png")
    Image.fromarray(filled_rgb).save(output_dir + "/5_filled_rgb.png")
    Image.fromarray(mask_np).save(output_dir + "/6_final_mask.png")

    return init_image, mask_np, refined_render_mask, rendered_depth


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
        guidance_scale: float = 3.5,
        use_rf_solver_invert: bool = False,
        use_rf_solver_sample: bool = False,
        source_key_scale: float = 1.0,
        source_value_scale: float = 1.0,
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
            gamma=1.0,
            use_rf_solver_invert=use_rf_solver_invert,
        )
        
        set_flux_transformer_attn_processor(
            self.pipe.transformer,
            set_attn_proc_func=lambda name, dh, nh, ap: PersonalizeAnythingAttnProcessor(
                name=name,
                tau=tau / 100,
                mask=mask,
                device=self.device,
                img_dims=img_dims,
                source_key_scale=source_key_scale,
                source_value_scale=source_value_scale,
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
            guidance_scale=guidance_scale,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            mask=mask,
            use_timestep=True,
            use_rf_solver_sample=use_rf_solver_sample,
        ).images[1]
        
        return image
