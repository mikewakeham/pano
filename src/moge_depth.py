# Auto-patch utils3d for numpy < 2.0 compatibility (one-time)
def _ensure_utils3d_patched():
    try:
        import utils3d
        from pathlib import Path
        import re
        
        transforms_file = Path(utils3d.__file__).parent / 'numpy' / 'transforms.py'
        
        if not transforms_file.exists():
            return  # File doesn't exist, skip
        
        content = transforms_file.read_text()
        
        # Check if already patched (no .mT present)
        if '.mT' not in content:
            return  # Already patched
        
        # Patch: replace .mT with np.swapaxes(..., -2, -1)
        pattern = r'(\S+)\.mT\b'
        patched = re.sub(pattern, r'np.swapaxes(\1, -2, -1)', content)
        
        transforms_file.write_text(patched)
        print(f"Auto-patched utils3d for numpy < 2.0 compatibility")
        
    except Exception as e:
        print(f"Warning: Could not auto-patch utils3d: {e}")
        print("You may need to manually patch or upgrade numpy.")

_ensure_utils3d_patched()

# Now continue with normal imports
import third_party
import numpy as np
from PIL import Image
from typing import Union
import cv2
import torch
import utils3d
from moge.model.v2 import MoGeModel
from moge.utils.panorama import get_panorama_cameras, split_panorama_image, merge_panorama_depth


class MoGe:
    def __init__(self, pretrained: str = "Ruicheng/moge-2-vitl", device: str = None, batch_size: int = 4):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        self.model = MoGeModel.from_pretrained(pretrained).to(self.device).eval()

    def __call__(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        if isinstance(image, Image.Image):
            image_np = np.array(image)
            if image_np.ndim == 3 and image_np.shape[2] == 4:
                image_np = image_np[:, :, :3]
        else:
            image_np = np.asarray(image).copy()
        
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        
        image = image_np if image_np.shape[2] == 3 else image_np[:, :, :3]
        height, width = image.shape[:2]

        splitted_extrinsics, splitted_intriniscs = get_panorama_cameras()
        splitted_resolution = 512
        splitted_images = split_panorama_image(image, splitted_extrinsics, splitted_intriniscs, splitted_resolution)

        splitted_distance_maps, splitted_masks = [], []
        for i in range(0, len(splitted_images), self.batch_size):
            image_tensor = torch.tensor(
                np.stack(splitted_images[i:i + self.batch_size]) / 255, 
                dtype=torch.float32, 
                device=self.device
            ).permute(0, 3, 1, 2)
            
            fov_x, fov_y = np.rad2deg(
                utils3d.np.intrinsics_to_fov(
                    np.array(splitted_intriniscs[i:i + self.batch_size])
                )
            )
            fov_x = torch.tensor(fov_x, dtype=torch.float32, device=self.device)
            
            with torch.no_grad():
                output = self.model.infer(image_tensor, fov_x=fov_x, apply_mask=False)
            
            distance_map = output['points'].norm(dim=-1).cpu().numpy()
            mask = output['mask'].cpu().numpy()
            splitted_distance_maps.extend(list(distance_map))
            splitted_masks.extend(list(mask))

        merging_width, merging_height = min(1920, width), min(960, height)
        panorama_depth, panorama_mask = merge_panorama_depth(
            merging_width, 
            merging_height, 
            splitted_distance_maps, 
            splitted_masks, 
            splitted_extrinsics, 
            splitted_intriniscs
        )
        
        panorama_depth = panorama_depth.astype(np.float32)
        panorama_depth = cv2.resize(panorama_depth, (width, height), cv2.INTER_LINEAR)
        
        return panorama_depth