import sys
import gc
import torch
from pathlib import Path
from PIL import Image
import gsplat
import time
import numpy as np

_parent_dir = Path(__file__).resolve().parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from hunyuanworld_mirror import HunyuanWorldMirror
from dit360 import DiT360, prepare_dit360_inputs
from dap import DAP
from update_3dgs import update_3dgs_with_outpaint
from render import render_cubemap, cubemap_frames_to_equirectangular, c2w_to_w2c
from moviepy.editor import ImageSequenceClip

def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

if __name__ == '__main__':
    input_fps=2
    hunyuan_model = HunyuanWorldMirror(fps=input_fps)
    predictions = hunyuan_model("/projectnb/ivc-ml/mwakeham/panorama/assets/library/library.mp4")
    print("hunyuanworld-mirror prediction done.")
    
    init_image, mask_np, rendered_mask, rendered_depth = prepare_dit360_inputs(
        predictions, frame_idx=0, confidence_threshold=0.9, subsample_ratio=0.2
    )
    print("dit360 inputs ready.")

    hunyuan_model.model = hunyuan_model.model.cpu()
    clear_gpu_memory()
    
    dit360_model = DiT360()
    prompt = "This is a panorama image. Inside an old, spacious large medieval library, standing in the center surrounded by wooden bookshelves at the circular wall. Warm sunlight fills the room with the worn wooden floor. Dust hangs in the still air, photorealistic, natural perspective from the middle of the old, worn down large room."
    
    outpainted_image = dit360_model(
        init_image, mask_np, prompt, timestep=50, seed=0, tau=25
    )
    del dit360_model
    print("outpainting complete!")

    output_path = Path("/projectnb/ivc-ml/mwakeham/panorama/assets/library/outpainted.png")
    outpainted_image.save(output_path)
    print(f"saved outpainted image to {output_path}")

    dap_model = DAP(model_path="/projectnb/ivc-ml/mwakeham/panorama/third_party/DAP/weights/model.pth")
    depth_map = dap_model(outpainted_image)
    print("panoramic depth estimation done.")

    updated_splats = update_3dgs_with_outpaint(
        hunyuan_predictions=predictions,
        outpainted_image=outpainted_image,
        dap_depth=depth_map,
        rendered_mask=mask_np,
        rendered_depth=rendered_depth,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        depth_align_percentile=90.0,
    )

    splats_output_path = Path("/projectnb/ivc-ml/mwakeham/panorama/assets/library/updated_splats.ply")
    gsplat.export_splats(
        means=updated_splats["means"],
        sh0=updated_splats["sh0"],
        shN=updated_splats["shN"],
        scales=updated_splats["scales"],
        quats=updated_splats["quats"],
        opacities=updated_splats["opacities"],
        format="ply",
        save_to=str(splats_output_path),
    )
    print(f"saved updated splats to {splats_output_path}")

    camera_poses_c2w = predictions['camera_poses'][0].cpu().numpy()
    camera_poses_w2c = np.stack([c2w_to_w2c(p) for p in camera_poses_c2w], axis=0)

    render_fps = 10
    cubemap_frames = render_cubemap(
        splats=updated_splats,
        w2c_poses=camera_poses_w2c,
        num_frames=int(predictions['camera_poses'].shape[1]*(render_fps/input_fps)),
        face_resolution=512,
    )
    print(f"cubemap frames shape: {cubemap_frames.shape}")

    equirectangular_frames = cubemap_frames_to_equirectangular(
        cubemap_frames=cubemap_frames, 
        equi_height=1024, 
        equi_width=2048
    )
    print(f"equirectangular frames shape: {equirectangular_frames.shape}")

    clip = ImageSequenceClip(list(equirectangular_frames), fps=render_fps)
    clip.write_videofile("/projectnb/ivc-ml/mwakeham/panorama/assets/library/equirectangular_render.mp4", codec="libx264")