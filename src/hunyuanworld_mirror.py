import torch
from pathlib import Path
from typing import Optional, Dict, Any, Union

from third_party import WorldMirror, extract_load_and_preprocess_images

class HunyuanWorldMirror:
    def __init__(
        self,
        model_name: str = "tencent/HunyuanWorld-Mirror",
        device: Optional[str] = None,
        fps: int = 1,
        target_size: int = 518,
    ):

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.fps = fps
        self.target_size = target_size
        
        print(f"loading model {model_name}...")
        self.model = WorldMirror.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("HunyuanWorld loaded")
    
    def __call__(
        self,
        input_path: Union[str, Path],
        camera_poses: Optional[torch.Tensor] = None,
        depthmap: Optional[torch.Tensor] = None,
        camera_intrs: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            input_path: Path to video file or directory of images
            camera_poses: Optional camera pose tensor [1, N, 4, 4]
            depthmap: Optional depth map tensor [1, N, H, W]
            camera_intrs: Optional camera intrinsics tensor [1, N, 3, 3]
            
        Returns:
            Dictionary with predictions containing:
            - pts3d: 3D point cloud [S, H, W, 3]
            - pts3d_conf: Point confidence [S, H, W]
            - depth: Depth predictions [S, H, W, 1]
            - depth_conf: Depth confidence [S, H, W]
            - normals: Surface normals [S, H, W, 3]
            - normals_conf: Normal confidence [S, H, W]
            - camera_poses: Camera poses [S, 4, 4]
            - camera_intrs: Camera intrinsics [S, 3, 3]
            - camera_params: Camera parameters [S, 9]
            - splats: 3D Gaussian Splatting outputs
            - input_images: Processed input images [1, N, 3, H, W]
        """

        inputs = {}
        input_images = extract_load_and_preprocess_images(
            Path(input_path),
            fps=self.fps,
            target_size=self.target_size
        ).to(self.device)
        inputs['img'] = input_images
        
        cond_flags = [0, 0, 0]  # [camera_pose, depth, intrinsics]
        if camera_poses is not None:
            cond_flags[0] = 1
            inputs['camera_poses'] = camera_poses.to(self.device)
        if depthmap is not None:
            cond_flags[1] = 1
            inputs['depthmap'] = depthmap.to(self.device)
        if camera_intrs is not None:
            cond_flags[2] = 1
            inputs['camera_intrs'] = camera_intrs.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(views=inputs, cond_flags=cond_flags)
        
        predictions['input_images'] = input_images
        
        return predictions