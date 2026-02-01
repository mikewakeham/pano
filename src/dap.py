import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Optional, Union
import os

from third_party import _dap_root
from networks.models import make
from argparse import Namespace

class DAP:
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        midas_model_type: str = "vitl",
        min_depth: float = 0.01,
        max_depth: float = 1.0,
    ):

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self.device)
        
        if model_path is None:
            raise ValueError("must pass model path for DAP")
        
        model_config = {
            'name': 'dap',
            'args': {
                'midas_model_type': midas_model_type,
                'fine_tune_type': 'hypersim',
                'min_depth': min_depth,
                'max_depth': max_depth,
                'train_decoder': True,
            }
        }
        
        original_cwd = os.getcwd()
        try:
            os.chdir(str(_dap_root))
            self.model = make(model_config)
        finally:
            os.chdir(original_cwd)
        
        state = torch.load(model_path, map_location=self.device)
        
        if any(k.startswith("module") for k in state.keys()):
            self.model = nn.DataParallel(self.model)
        
        print("loading DAP model...")
        self.model = self.model.to(self.device)
        m_state = self.model.state_dict()
        self.model.load_state_dict(
            {k: v for k, v in state.items() if k in m_state}, 
            strict=False
        )
        self.model.eval()
        
        print("DAP model loaded")
    
    def __call__(
        self,
        image: Union[Image.Image, np.ndarray],
        input_size: int = 518,
    ) -> np.ndarray:

        if isinstance(image, Image.Image):
            image_np = np.array(image)
            if image_np.shape[2] == 4:
                image_np = image_np[:, :, :3]
        else:
            image_np = image.copy()
        
        if image_np.dtype != np.uint8:
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)
        
        import cv2
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        with torch.no_grad():
            if hasattr(self.model, 'module'):
                depth = self.model.module.infer_image(image_bgr, input_size=input_size)
            else:
                depth = self.model.infer_image(image_bgr, input_size=input_size)
        
        return depth

