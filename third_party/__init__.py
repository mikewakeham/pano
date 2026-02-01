import sys
from pathlib import Path

_module_dir = Path(__file__).parent

_hunyuan_root = _module_dir / "HunyuanWorld-Mirror"
if str(_hunyuan_root) not in sys.path:
    sys.path.insert(0, str(_hunyuan_root))

from src.models.models.worldmirror import WorldMirror
from src.utils.inference_utils import prepare_images_to_tensor, VIDEO_EXTS, IMAGE_EXTS
from src.utils.video_utils import video_to_image_frames
import tempfile
import glob
import os
import importlib
import types

def extract_load_and_preprocess_images(image_folder_or_video_path, fps=1, target_size=518, mode="crop"):
    from pathlib import Path
    path = Path(image_folder_or_video_path)
    
    if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
        temp_dir = Path(tempfile.mkdtemp(prefix="hunyuan_frames_"))
        frame_paths = video_to_image_frames(str(path), str(temp_dir), fps=fps)
        img_paths = sorted(frame_paths)
    else:
        img_paths = []
        for ext in IMAGE_EXTS:
            img_paths.extend(glob.glob(os.path.join(str(path), ext)))
        img_paths = sorted(img_paths)
    images = prepare_images_to_tensor(img_paths, resize_strategy=mode, target_size=target_size)
    return images

def _create_submodule(namespace, name):
    module = types.ModuleType(f"{namespace}.{name}")
    src_module = importlib.import_module(f"src.{name}")
    module.__dict__.update(src_module.__dict__)
    sys.modules[f"{namespace}.{name}"] = module
    return module

_hunyuan_models = _create_submodule("third_party.hunyuanworld", "models")
_hunyuan_utils = _create_submodule("third_party.hunyuanworld", "utils")
sys.modules["third_party.hunyuanworld.models"] = _hunyuan_models
sys.modules["third_party.hunyuanworld.utils"] = _hunyuan_utils

_dit360_root = _module_dir / "DiT360"
if str(_dit360_root) not in sys.path:
    sys.path.insert(0, str(_dit360_root))

_image_outpaint_module = types.ModuleType("image_outpaint")
sys.modules["image_outpaint"] = _image_outpaint_module

_dit360_submodule = types.ModuleType("image_outpaint.DiT360")
sys.modules["image_outpaint.DiT360"] = _dit360_submodule

_pa_src_module = types.ModuleType("image_outpaint.DiT360.pa_src")
sys.modules["image_outpaint.DiT360.pa_src"] = _pa_src_module
_dit360_submodule.pa_src = _pa_src_module

try:
    import importlib.util
    for module_name in ['pipeline', 'attn_processor', 'utils']:
        module_path = _dit360_root / "pa_src" / f"{module_name}.py"
        if module_path.exists():
            spec = importlib.util.spec_from_file_location(
                f"image_outpaint.DiT360.pa_src.{module_name}",
                module_path,
                submodule_search_locations=[str(_dit360_root / "pa_src")]
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[f"image_outpaint.DiT360.pa_src.{module_name}"] = module
                spec.loader.exec_module(module)
                setattr(_pa_src_module, module_name, module)
except Exception as e:
    pass

_dap_root = _module_dir / "DAP"
if str(_dap_root) not in sys.path:
    sys.path.insert(0, str(_dap_root))

__all__ = ["WorldMirror", "extract_load_and_preprocess_images", "_dap_root"]

