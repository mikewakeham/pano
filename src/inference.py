import sys
import gc
import torch
import argparse
from pathlib import Path
from PIL import Image
import gsplat
import time
import numpy as np
import cv2
from tqdm import tqdm

_parent_dir = Path(__file__).resolve().parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

import third_party
from hunyuanworld_mirror import HunyuanWorldMirror
from dit360 import DiT360, prepare_dit360_inputs, find_least_covered_frame_idx
from mask_utils import fill_video_sequence_telea
from update_3dgs import update_3dgs_with_outpaint, convert_hunyuanworld_to_gsplat
from render import (
    render_panoramic_video, cubemap_frames_to_equirectangular,
    cubemap_alpha_to_equirectangular_mask, cubemap_depth_to_equirectangular, c2w_to_w2c,
    get_interpolated_camera_poses, prune_reference_view_floaters
)
from ttt_wan_vace import refine_panoramic_video
from moviepy.editor import ImageSequenceClip, VideoFileClip


def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Input/Output paths
    parser.add_argument("--input_video", type=str, required=True,
                        help="Path to input video file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save all outputs")
    
    # Outpainting options
    outpaint_group = parser.add_mutually_exclusive_group(required=True)
    outpaint_group.add_argument("--outpainted_image_path", type=str,
                               help="Path to pre-existing outpainted image (skips DiT360 generation)")
    outpaint_group.add_argument("--scene_prompt", type=str,
                               help="Text prompt for scene outpainting with DiT360")
    
    # Depth estimation
    parser.add_argument("--depth_model", type=str, default="moge", choices=["dap", "moge"],
                        help="Depth estimator: dap or moge")
    parser.add_argument("--dap_weights", type=str,
                        default="/projectnb/ivc-ml/mwakeham/panorama/third_party/DAP/weights/model.pth",
                        help="Path to DAP model weights (used when depth_model=dap)")
    parser.add_argument("--moge_pretrained", type=str, default="Ruicheng/moge-2-vitl",
                        help="MoGe pretrained model (used when depth_model=moge)")
    
    # Video and rendering parameters
    parser.add_argument("--input_fps", type=int, default=2,
                        help="FPS for input video processing")
    parser.add_argument("--render_fps", type=int, default=10,
                        help="FPS for final rendered output")
    parser.add_argument("--frame_idx", type=int, default=-1,
                    help="Frame index for outpainting initialization. -1 = auto-select least covered.")
    
    # DiT360 parameters
    parser.add_argument("--confidence_threshold", type=float, default=0.9,
                        help="Confidence threshold for prepare_dit360_inputs")
    parser.add_argument("--subsample_ratio", type=float, default=1.0,
                        help="Subsample ratio for prepare_dit360_inputs")
    parser.add_argument("--dit360_timestep", type=int, default=50,
                        help="Number of timesteps for DiT360 diffusion")
    parser.add_argument("--dit360_seed", type=int, default=0,
                        help="Random seed for DiT360")
    parser.add_argument("--dit360_tau", type=float, default=10.0,
                        help="Tau parameter for DiT360")
    parser.add_argument("--dit360_guidance_scale", type=float, default=2.5,
                        help="Guidance scale for DiT360")
    parser.add_argument("--dit360_use_rf_solver_invert", action="store_true",
                        help="Use midpoint second-order RF-Solver updates during inversion")
    parser.add_argument("--dit360_use_rf_solver_sample", action="store_true",
                        help="Use midpoint second-order RF-Solver updates during sampling")
    parser.add_argument("--dit360_source_key_scale", type=float, default=1.09,
                        help="Scale factor for source-region keys in the edited branch attention.",)
    parser.add_argument("--dit360_source_value_scale", type=float, default=1.0,
                        help="Scale factor for source-region values in the edited branch attention.",)
    
    # Rendering parameters
    parser.add_argument("--render_type", type=str, default="cubemap",
                        choices=["cubemap", "icosahedron"],
                        help="Rendering method: cubemap (6 faces) or icosahedron")
    parser.add_argument("--face_resolution", type=int, default=1024,
                        help="Resolution for cubemap faces")
    parser.add_argument("--equi_height", type=int, default=1024,
                        help="Height of equirectangular output")
    parser.add_argument("--equi_width", type=int, default=2048,
                        help="Width of equirectangular output")
    parser.add_argument("--alpha_threshold", type=float, default=0.8,
                        help="Alpha threshold for mask conversion")
    parser.add_argument("--blend_mode", type=str, default="weighted",
                        choices=["weighted", "nearest"],
                        help="Blending mode for icosahedron (ignored for cubemap)")
    
    # 3DGS update parameters
    parser.add_argument("--depth_align_percentile", type=float, default=90.0,
                        help="Percentile for depth alignment")
    parser.add_argument("--save_splats", action="store_true",
                        help="Save updated 3D Gaussian Splats to .ply file")
    
    # Optional interpolation
    parser.add_argument("--fill_video_telea", action="store_true",
                        help="Apply telea inpainting to rendered video")
    parser.add_argument("--fill_telea_workers", type=int, default=14,
                        help="Parallel processes for Telea fill (1 = sequential). Only with --fill_video_telea")

    # TTT Wan VACE refinement parameters
    parser.add_argument("--refine_with_vace", action="store_true",
                        help="Apply TTT refinement using Wan 2.1 VACE")
    parser.add_argument("--vace_checkpoint_dir", type=str,
                        default="/projectnb/ivc-ml/mwakeham/panorama/checkpoints/Wan2.1-VACE-1.3B",
                        help="Path to Wan2.1 VACE checkpoint directory")
    parser.add_argument("--vace_model_size", type=str, default="1.3B",
                        choices=["1.3B", "7B"],
                        help="VACE model size")
    parser.add_argument("--vace_num_train_steps", type=int, default=None,
                        help="Number of TTT training steps for VACE. If not set or 0, inference only (no training)")
    parser.add_argument("--vace_learning_rate", type=float, default=2e-5,
                        help="Learning rate for VACE TTT")
    parser.add_argument("--vace_lora_rank", type=int, default=64,
                        help="LoRA rank for VACE TTT")
    parser.add_argument("--vace_sampling_steps", type=int, default=50,
                        help="Number of sampling steps for VACE")
    parser.add_argument("--vace_guide_scale", type=float, default=5.0,
                        help="Guidance scale for VACE")
    parser.add_argument("--vace_shift", type=float, default=5.0,
                        help="Shift parameter for VACE")
    parser.add_argument("--vace_seed", type=int, default=42,
                        help="Random seed for VACE")
    parser.add_argument("--save_vace_lora", action="store_true",
                        help="Save VACE LoRA checkpoint after training")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Process input video with HunyuanWorld
    print(f"Processing input video: {args.input_video}")
    hunyuan_model = HunyuanWorldMirror(fps=args.input_fps)
    predictions = hunyuan_model(args.input_video)
    print("HunyuanWorld-Mirror prediction done.")

    if args.save_splats:
        gsplat_splats = convert_hunyuanworld_to_gsplat(predictions["splats"])
        gsplat.export_splats(
            means=gsplat_splats["means"],
            sh0=gsplat_splats["sh0"],
            shN=gsplat_splats["shN"],
            scales=gsplat_splats["scales"],
            quats=gsplat_splats["quats"],
            opacities=gsplat_splats["opacities"],
            format="ply",
            save_to=output_dir / "input_splats.ply",
        )
        print("saved HunyuanWorld splats to input_splats.ply")
    
    # Interpolate camera poses
    camera_poses_c2w_sparse = predictions['camera_poses'][0].cpu().numpy()
    interpolated_c2w, interpolated_w2c, num_render_frames = get_interpolated_camera_poses(
        video_path=args.input_video,
        camera_poses_c2w=camera_poses_c2w_sparse,
        target_fps=args.render_fps,
    )
    print(f"Interpolated {len(camera_poses_c2w_sparse)} sparse poses to {num_render_frames} frames")
    
    if args.frame_idx == -1:
        frame_idx = find_least_covered_frame_idx(predictions, interpolated_c2w)
    else:
        frame_idx = args.frame_idx

    # Prepare DiT360 inputs
    init_image, mask_np, rendered_mask, rendered_depth = prepare_dit360_inputs(
        predictions, 
        frame_idx=frame_idx, 
        interpolated_camera_poses=interpolated_c2w,
        confidence_threshold=args.confidence_threshold, 
        subsample_ratio=args.subsample_ratio
    )
    init_image.save(output_dir / "refined_rgb_input.png")
    Image.fromarray(mask_np).save(output_dir / "refined_mask_input.png")
    print("DiT360 inputs ready.")
    
    # Free up GPU memory
    hunyuan_model.model = hunyuan_model.model.cpu()
    clear_gpu_memory()
    
    # Generate or load outpainted image
    if args.outpainted_image_path is not None:
        print(f"Loading pre-existing outpainted image from: {args.outpainted_image_path}")
        outpainted_image = Image.open(args.outpainted_image_path)
    else:
        print("Generating outpainted image with DiT360...")
        dit360_model = DiT360()
        outpainted_image = dit360_model(
            init_image, 
            mask_np, 
            args.scene_prompt, 
            timestep=args.dit360_timestep, 
            seed=args.dit360_seed, 
            tau=args.dit360_tau, 
            guidance_scale=args.dit360_guidance_scale,
            use_rf_solver_invert=args.dit360_use_rf_solver_invert,
            use_rf_solver_sample=args.dit360_use_rf_solver_sample,
            source_key_scale=args.dit360_source_key_scale,
            source_value_scale=args.dit360_source_value_scale,
        )
        del dit360_model
        clear_gpu_memory()
    print("Outpainting complete!")
    
    # Save outpainted image
    outpainted_path = output_dir / "outpainted.png"
    outpainted_image.save(outpainted_path)
    print(f"Saved outpainted image to {outpainted_path}")
    
    # Estimate panoramic depth
    print("Estimating panoramic depth...")
    if args.depth_model == "dap":
        from dap import DAP
        dap_model = DAP(model_path=args.dap_weights)
        depth_map, depth_valid_mask = dap_model(outpainted_image)
        _vis = cv2.applyColorMap(((depth_map - depth_map.min()) / max(depth_map.max() - depth_map.min(), 1e-6) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        _vis[~depth_valid_mask] = _vis.max()
        cv2.imwrite(str(output_dir / "dap_depth_vis.png"), _vis)
    else:
        from moge_depth import MoGe
        moge_model = MoGe(pretrained=args.moge_pretrained)
        depth_map, depth_valid_mask = moge_model(outpainted_image)
        _vis = cv2.applyColorMap(((depth_map - depth_map.min()) / max(depth_map.max() - depth_map.min(), 1e-6) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        _vis[~depth_valid_mask] = _vis.max()
        cv2.imwrite(str(output_dir / "moge_depth_vis.png"), _vis)
    print("Panoramic depth estimation done.")
    
    # Update 3D Gaussian Splats
    print("Updating 3D Gaussian Splats...")
    base_splats = convert_hunyuanworld_to_gsplat(predictions["splats"])
    base_splats = prune_reference_view_floaters(
        base_splats=base_splats,
        w2c_pose=interpolated_w2c[frame_idx],
        equi_height=args.equi_height,
        equi_width=args.equi_width,
        face_resolution=args.face_resolution,
        device=device,
        radius_percentile=99.0,
        max_other_overlap_count=10,
        low_support_fraction_threshold=0.1,
    )

    updated_splats = update_3dgs_with_outpaint(
        base_splats=base_splats,
        outpainted_image=outpainted_image,
        depth=depth_map,
        rendered_mask=rendered_mask,
        rendered_depth=rendered_depth,
        frame_idx=frame_idx,
        interpolated_camera_poses=interpolated_c2w,
        depth_valid_mask=depth_valid_mask,
        device=device,
        depth_align_percentile=args.depth_align_percentile,
    )
    
    if args.save_splats:
        splats_path = output_dir / "updated_splats.ply"
        gsplat.export_splats(
            means=updated_splats["means"],
            sh0=updated_splats["sh0"],
            shN=updated_splats["shN"],
            scales=updated_splats["scales"],
            quats=updated_splats["quats"],
            opacities=updated_splats["opacities"],
            format="ply",
            save_to=str(splats_path),
        )
        print(f"Saved updated splats to {splats_path}")
    
    # Render cubemap
    print(f"Rendering equirectangular video using {args.render_type}...")
    equirectangular_frames, equirectangular_masks, equirectangular_depth = render_panoramic_video(
        splats=updated_splats,
        w2c_poses=interpolated_w2c,
        num_frames=num_render_frames,
        render_type=args.render_type,
        face_resolution=args.face_resolution,
        equi_height=args.equi_height,
        equi_width=args.equi_width,
        alpha_threshold=None,
        blend_mode=args.blend_mode,
        device=device,
    )
    
    equirectangular_binary_mask = equirectangular_masks > args.alpha_threshold
    
    render_video_path = output_dir / "equirectangular_render.mp4"
    ImageSequenceClip(list(equirectangular_frames), fps=args.render_fps).write_videofile(
        str(render_video_path), codec="libx264", logger=None
    )
    print(f"Saved render video to {render_video_path}")
    
    mask_video_path = output_dir / "equirectangular_mask.mp4"
    mask_frames = (equirectangular_binary_mask.astype(np.uint8) * 255)[..., None].repeat(3, axis=-1)
    ImageSequenceClip(list(mask_frames), fps=args.render_fps).write_videofile(
        str(mask_video_path), codec="libx264", logger=None
    )
    print(f"Saved mask video to {mask_video_path}")

    alpha_video_path = output_dir / "equirectangular_alpha.mp4"
    alpha_frames = (equirectangular_masks * 255).astype(np.uint8)[..., None].repeat(3, axis=-1)
    ImageSequenceClip(list(alpha_frames), fps=args.render_fps).write_videofile(
        str(alpha_video_path), codec="libx264", logger=None
    )
    print(f"Saved alpha video to {alpha_video_path}")

    if args.fill_video_telea:
        filled_frames = fill_video_sequence_telea(
            equirectangular_frames,
            equirectangular_masks,
            num_workers=args.fill_telea_workers,
        )
        filled_video_path = output_dir / "equirectangular_filled.mp4"
        ImageSequenceClip(list(filled_frames), fps=args.render_fps).write_videofile(
            str(filled_video_path), codec="libx264", logger=None
        )
        print(f"Saved filled video to {filled_video_path}")
    
    depth_gray = (255 * np.clip(
        (equirectangular_depth - np.percentile(equirectangular_depth[equirectangular_depth > 0], 2)) / 
        max(np.percentile(equirectangular_depth[equirectangular_depth > 0], 98) - 
            np.percentile(equirectangular_depth[equirectangular_depth > 0], 2), 1e-6), 
        0, 1
    )).astype(np.uint8)
    depth_vis = np.stack([cv2.applyColorMap(frame, cv2.COLORMAP_TURBO)[..., ::-1] for frame in depth_gray], axis=0)
    depth_video_path = output_dir / "equirectangular_depth.mp4"
    ImageSequenceClip(list(depth_vis), fps=args.render_fps).write_videofile(
        str(depth_video_path), codec="libx264", logger=None
    )
    print(f"Saved depth video to {depth_video_path}")

    if args.refine_with_vace:
        print("Inpainting/refining with Wan VACE")
        clear_gpu_memory()
        
        save_lora_path = None
        if args.save_vace_lora:
            save_lora_path = str(output_dir / f"vace_lora_checkpoint_{args.vace_num_train_steps}")
        
        vace_prompt = args.scene_prompt if args.scene_prompt is not None else ""
        
        refined_frames = refine_panoramic_video(
            video_frames=equirectangular_frames,
            mask_frames=equirectangular_masks,
            checkpoint_dir=args.vace_checkpoint_dir,
            prompt=vace_prompt,
            device=device,
            model_size=args.vace_model_size,
            num_train_steps=args.vace_num_train_steps,
            learning_rate=args.vace_learning_rate,
            lora_rank=args.vace_lora_rank,
            sampling_steps=args.vace_sampling_steps,
            guide_scale=args.vace_guide_scale,
            shift=args.vace_shift,
            seed=args.vace_seed,
            save_lora_path=save_lora_path,
        )
        print(f"Refined frames shape: {refined_frames.shape}")
        
        # Save refined video
        n = args.vace_num_train_steps if args.vace_num_train_steps else "inference"
        refined_video_path = output_dir / f"equirectangular_inpainted_{n}.mp4"
        clip = ImageSequenceClip(list(refined_frames[:num_render_frames]), fps=args.render_fps)
        clip.write_videofile(str(refined_video_path), codec="libx264", logger=None)
        print(f"Saved refined video to {refined_video_path}")

if __name__ == '__main__':
    main()
