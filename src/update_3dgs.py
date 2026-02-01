import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any
from PIL import Image
from einops import rearrange
import gsplat
import os.path as osp

def sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    indices = q_abs.argmax(dim=-1, keepdim=True)
    expand_dims = list(batch_dim) + [1, 4]
    gather_indices = indices.unsqueeze(-1).expand(expand_dims)
    out = torch.gather(quat_candidates, -2, gather_indices).squeeze(-2)
    return standardize_quaternion(out)


def equi_unit_rays(h: int, w: int, device):
    u = (torch.arange(w, device=device).float() + 0.5) / w
    v = (torch.arange(h, device=device).float() + 0.5) / h
    vv, uu = torch.meshgrid(v, u, indexing="ij")   # (H,W) order here

    phi   = uu * 2 * torch.pi - torch.pi  # [-pi, +pi]
    theta = torch.pi / 2 - vv * torch.pi  # [pi/2, -pi/2]

    x = - torch.cos(theta) * torch.sin(phi)
    y = torch.sin(theta)
    z = torch.cos(theta) * torch.cos(phi)
    return torch.stack((x, y, z), dim=-1)  # (h,w,3)


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def convert_rgbd_equi_to_3dgs(
    rgb: torch.Tensor,                       # (H, W, 3) RGB image
    distance: torch.Tensor,                  # (H, W) Distance map
    rays: Optional[torch.Tensor] = None,     # (H, W, 3) Ray directions (unit vectors ideally)
    mask: Optional[torch.Tensor] = None,     # (H, W) Optional boolean mask
    dis_threshold=0.,
    epsilon=1e-3,
    scale_rate=1.0,
    save_path: Optional[str] = None,
) -> nn.ParameterDict:
    """
    Given an equirectangular RGB-D image, back-project each pixel to a 3D point
    and compute the corresponding 3D Gaussian covariance so that the projection covers 1 pixel.

    Returns:
        centers (N x 3): 3D positions of splats
        covariances (N x 3 x 3): 3D Gaussian covariances of splats
        colors (N x 3): RGB values of splats
        opacities (N x 1): Opacities of splats
        scales (N x 3): Scales of splats
        rotations (N x 4): Rotations of splats
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3, "Image must be HxWx3"
    assert distance.ndim == 2, "Distance must be HxW"
    assert rgb.shape[:2] == distance.shape[:2], "Input shapes must match"
    if rays is not None:
        assert rgb.shape[:2] == rays.shape[:2], "Input shapes must match"
        assert rays.ndim == 3 and rays.shape[2] == 3, "Rays must be HxWx3"
    if mask is not None:
        assert mask.ndim == 2 and mask.shape[:2] == rgb.shape[:2], "Mask shape must match"
        assert mask.dtype == torch.bool, "Mask must be a boolean tensor"

    H, W = rgb.shape[:2]
    device = rgb.device

    # Normalize RGB to [0, 1] if it's in [0, 255] range
    if rgb.max() > 1.0:
        rgb = rgb / 255.0

    if rays is None:
        rays = equi_unit_rays(rgb.shape[0], rgb.shape[1], device)
        rays[..., [0, 1]] *= -1
    
    valid_mask = distance > dis_threshold
    if mask is not None:
        valid_mask = valid_mask & mask.unsqueeze(0).unsqueeze(0)
    rays_flat = rays.view(-1, 3)
    rgbs_flat = rgb.view(-1, 3)
    distance_flat = distance.view(-1)
    valid_rays = rays_flat[valid_mask.view(-1)]
    valid_rgbs = rgbs_flat[valid_mask.view(-1)]
    valid_distance = distance_flat[valid_mask.view(-1)]
    centers = valid_rays * valid_distance[:, None]

    delta_phi = 2 * torch.pi / rgb.shape[1]
    delta_theta = torch.pi / rgb.shape[0]
    sigma_x = valid_distance * delta_phi * scale_rate
    sigma_y = valid_distance * delta_theta * scale_rate
    sigma_z = torch.ones_like(valid_distance) * epsilon * scale_rate

    S = torch.stack([sigma_x, sigma_y, sigma_z], dim=1)

    up = torch.tensor([0, 1, 0], dtype=torch.float32, device=device).expand_as(valid_rays)
    x_axis = torch.nn.functional.normalize(torch.cross(up, valid_rays, dim=-1), dim=1)
    fallback_up = torch.tensor([1, 0, 0], dtype=torch.float32, device=device).expand_as(valid_rays)
    degenerate_mask = torch.isnan(x_axis).any(dim=1)
    x_axis[degenerate_mask] = torch.nn.functional.normalize(torch.cross(fallback_up[degenerate_mask], valid_rays[degenerate_mask], dim=-1), dim=1)
    y_axis = torch.nn.functional.normalize(torch.cross(valid_rays, x_axis, dim=-1), dim=1)
    z_axis = valid_rays

    R = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # (N, 3, 3)

    sh_degree = 2
    num_sh_coeffs = (sh_degree + 1) ** 2  # 9 for degree 2

    # SH0 from RGB
    sh0 = rearrange(rgb_to_sh(valid_rgbs), 'n c -> n 1 c')  # (N, 1, 3)

    # Initialize remaining SH coefficients (1-8) to zero
    shN = torch.zeros((centers.shape[0], num_sh_coeffs - 1, 3), device=device)
    
    def inverse_sigmoid(x):
        return torch.log(x/(1-x))

    inverse_scaling_activation = torch.log    
    inverse_opacity_activation = inverse_sigmoid
    
    scales = inverse_scaling_activation(S)
    alphas = torch.ones((centers.shape[0],), device=device) * 0.99
    opacities = inverse_opacity_activation(alphas)
    
    quats = matrix_to_quaternion(R)

    splats = nn.ParameterDict({
        "means": nn.Parameter(centers, requires_grad=True),
        "sh0": nn.Parameter(sh0, requires_grad=True),
        "shN": nn.Parameter(shN, requires_grad=True),
        "scales": nn.Parameter(scales, requires_grad=True),
        "quats": nn.Parameter(quats, requires_grad=True),
        "opacities": nn.Parameter(opacities, requires_grad=True),
    })
    
    if save_path is not None:
        gsplat.export_splats(
            means=splats["means"],
            sh0=splats["sh0"],
            shN=splats["shN"],
            scales=splats["scales"],
            quats=splats["quats"],
            opacities=splats["opacities"],
            format=osp.splitext(save_path)[-1].lower().lstrip('.'),
            save_to=save_path,
        )
    
    return splats


def align_depth_scale(
    source_depth: np.ndarray,
    target_depth: np.ndarray,
    mask: np.ndarray,
    percentile: float = 60.0,
) -> float:
    """
    Align source depth to target depth using middle percentile median.
    
    Args:
        source_depth: Source depth map (DAP) [H, W]
        target_depth: Target depth map (HunyuanWorld rendered) [H, W]
        mask: Boolean mask for valid pixels [H, W]
        percentile: Percentile range to use (e.g., 60 = use middle 60%)
        
    Returns:
        scale: Scale factor to multiply source_depth
    """
    valid_source = np.where(mask, source_depth, np.nan)
    valid_target = np.where(mask, target_depth, np.nan)
    
    ratios = valid_target / valid_source
    q1, q2 = np.nanpercentile(ratios, [50 - (percentile / 2), 50 + (percentile / 2)])
    middle_mask = (ratios >= q1) & (ratios <= q2) & ~np.isnan(ratios)
    
    middle_source = valid_source[middle_mask]
    middle_target = valid_target[middle_mask]
    
    # Compute median ratio
    scale = np.nanmedian(middle_target / middle_source)
    
    print(f"Depth alignment: percentile range [{q1:.6f}, {q2:.6f}], scale: {scale:.6f}")
    
    return scale


def merge_splats(
    existing_splats: Dict[str, torch.Tensor],
    new_splats: nn.ParameterDict,
) -> nn.ParameterDict:
    """
    Merge new splats with existing HunyuanWorld splats, converting to gsplat format.
    
    Format conversion:
    - HunyuanWorld stores scales/opacities in activated form (exp/sigmoid applied)
    - gsplat expects scales/opacities in log/logit space (before activation)
    - HunyuanWorld uses 'sh' [N, K, 3], gsplat uses 'sh0' [N, 1, 3] + 'shN' [N, 8, 3]
    
    Args:
        existing_splats: HunyuanWorld splats dict with:
            - means: [B, N, 3] or [N, 3] - 3D positions
            - scales: [B, N, 3] or [N, 3] - scales in activated form (exp applied, clamped to max 0.3)
            - quats: [B, N, 4] or [N, 4] - quaternions (normalized)
            - opacities: [B, N] or [N] - opacities in activated form (sigmoid applied, in [0, 1])
            - sh: [B, N, K, 3] or [N, K, 3] - spherical harmonics (K=1 for degree 0, already in SH space)
        new_splats: New splats from convert_rgbd_equi_to_3dgs (already in gsplat format):
            - means: [N_new, 3]
            - scales: [N_new, 3] - in log space
            - quats: [N_new, 4]
            - opacities: [N_new] - in logit space (1D)
            - sh0: [N_new, 1, 3]
            - shN: [N_new, 8, 3]
            
    Returns:
        Merged ParameterDict in gsplat format:
            - means: [N_total, 3]
            - scales: [N_total, 3] - in log space
            - quats: [N_total, 4]
            - opacities: [N_total] - in logit space (1D)
            - sh0: [N_total, 1, 3] - first SH coefficient
            - shN: [N_total, 8, 3] - remaining SH coefficients (padded/truncated to 8)
    """
    # Handle batch dimension in existing splats
    if "means" in existing_splats:
        existing_means = existing_splats["means"]
        # Handle list input (HunyuanWorld sometimes returns lists)
        if isinstance(existing_means, list):
            existing_means = existing_means[0] if len(existing_means) > 0 else torch.tensor([], dtype=torch.float32)
        # Convert to tensor if not already
        if not isinstance(existing_means, torch.Tensor):
            existing_means = torch.tensor(existing_means)
        if existing_means.ndim == 3:
            existing_means = existing_means[0]  # [N, 3]
    else:
        raise ValueError("existing_splats must have 'means'")
    
    existing_scales = existing_splats["scales"]
    if isinstance(existing_scales, list):
        existing_scales = existing_scales[0] if len(existing_scales) > 0 else torch.tensor([], dtype=torch.float32)
    if not isinstance(existing_scales, torch.Tensor):
        existing_scales = torch.tensor(existing_scales)
    if existing_scales.ndim == 3:
        existing_scales = existing_scales[0]  # [N, 3]
    
    existing_quats = existing_splats["quats"]
    if isinstance(existing_quats, list):
        existing_quats = existing_quats[0] if len(existing_quats) > 0 else torch.tensor([], dtype=torch.float32)
    if not isinstance(existing_quats, torch.Tensor):
        existing_quats = torch.tensor(existing_quats)
    if existing_quats.ndim == 3:
        existing_quats = existing_quats[0]  # [N, 4]
    
    existing_opacities = existing_splats["opacities"]
    if isinstance(existing_opacities, list):
        existing_opacities = existing_opacities[0] if len(existing_opacities) > 0 else torch.tensor([], dtype=torch.float32)
    if not isinstance(existing_opacities, torch.Tensor):
        existing_opacities = torch.tensor(existing_opacities)
    # Handle batch dimension: [B, N] -> [N]
    if existing_opacities.ndim == 2:
        existing_opacities = existing_opacities[0]  # [N]
    # Ensure 1D: [N] (gsplat expects 1D opacities)
    if existing_opacities.ndim > 1:
        existing_opacities = existing_opacities.squeeze()
    if existing_opacities.ndim == 0:
        existing_opacities = existing_opacities.unsqueeze(0)
    
    # Handle SH coefficients
    if "sh" in existing_splats:
        existing_sh = existing_splats["sh"]
        if isinstance(existing_sh, list):
            existing_sh = existing_sh[0] if len(existing_sh) > 0 else torch.tensor([], dtype=torch.float32).reshape(0, 1, 3)
        if not isinstance(existing_sh, torch.Tensor):
            existing_sh = torch.tensor(existing_sh)
        # Handle batch dimension: [B, N, K, 3] -> [N, K, 3]
        if existing_sh.ndim == 4:
            existing_sh = existing_sh[0]  # [N, K, 3] or [N, 1, 3]
        # Ensure 3D: [N, K, 3]
        if existing_sh.ndim == 2:
            # If [N, 3], assume it's SH0 and reshape to [N, 1, 3]
            existing_sh = existing_sh.unsqueeze(1)  # [N, 1, 3]
        elif existing_sh.ndim == 1:
            # If [N*3], reshape to [N, 1, 3]
            existing_sh = existing_sh.reshape(-1, 1, 3)
        
        # HunyuanWorld typically uses sh[0] which is [N, 1, 3] for degree 0
        # Convert to sh0, shN format
        if existing_sh.shape[0] == 0:
            # Empty tensor - need device info
            device_sh = existing_sh.device if existing_sh.numel() > 0 else existing_means.device
            existing_sh0 = torch.zeros((0, 1, 3), device=device_sh, dtype=existing_sh.dtype)
            existing_shN = torch.zeros((0, 8, 3), device=device_sh, dtype=existing_sh.dtype)
        elif existing_sh.ndim == 3 and existing_sh.shape[1] == 1:
            # Only SH0, create shN with zeros
            existing_sh0 = existing_sh  # [N, 1, 3]
            existing_shN = torch.zeros((existing_sh0.shape[0], 8, 3), device=existing_sh0.device, dtype=existing_sh0.dtype)
        elif existing_sh.ndim == 3 and existing_sh.shape[1] > 1:
            # Has multiple SH coefficients
            existing_sh0 = existing_sh[:, 0:1, :]  # [N, 1, 3]
            existing_shN = existing_sh[:, 1:, :]  # [N, K-1, 3]
            # Pad or truncate to match standard 8 coefficients
            if existing_shN.shape[1] < 8:
                padding = torch.zeros((existing_shN.shape[0], 8 - existing_shN.shape[1], 3), device=existing_shN.device, dtype=existing_shN.dtype)
                existing_shN = torch.cat([existing_shN, padding], dim=1)
            elif existing_shN.shape[1] > 8:
                existing_shN = existing_shN[:, :8, :]
        else:
            # Unexpected shape - try to handle gracefully
            raise ValueError(f"Unexpected SH shape: {existing_sh.shape}, expected [N, K, 3] or [N, 1, 3]")
    else:
        # If no SH, create from colors or zeros
        if "colors" in existing_splats:
            existing_colors = existing_splats["colors"]
            if isinstance(existing_colors, list):
                existing_colors = existing_colors[0] if len(existing_colors) > 0 else torch.tensor([], dtype=torch.float32)
            if not isinstance(existing_colors, torch.Tensor):
                existing_colors = torch.tensor(existing_colors)
            if existing_colors.ndim == 3:
                existing_colors = existing_colors[0]
            # Convert RGB to SH0 (simplified - just use RGB as SH0)
            existing_sh0 = existing_colors.unsqueeze(1)  # [N, 1, 3]
            existing_shN = torch.zeros((existing_sh0.shape[0], 8, 3), device=existing_sh0.device)
        else:
            raise ValueError("existing_splats must have either 'sh' or 'colors'")
    
    # Extract new splats (already in gsplat format from convert_rgbd_equi_to_3dgs)
    new_means = new_splats["means"].data  # [N_new, 3]
    new_scales = new_splats["scales"].data  # [N_new, 3] - already in log space
    new_quats = new_splats["quats"].data  # [N_new, 4]
    new_opacities = new_splats["opacities"].data  # [N_new] - already in logit space (1D)
    new_sh0 = new_splats["sh0"].data  # [N_new, 1, 3]
    new_shN = new_splats["shN"].data  # [N_new, 8, 3]
    
    # Ensure all tensors are on same device
    device = existing_means.device
    new_means = new_means.to(device)
    new_scales = new_scales.to(device)
    new_quats = new_quats.to(device)
    new_opacities = new_opacities.to(device)
    new_sh0 = new_sh0.to(device)
    new_shN = new_shN.to(device)
    
    # Convert HunyuanWorld's activated values to gsplat's log/logit space
    # HunyuanWorld stores scales/opacities in activated form (exp/sigmoid applied)
    # gsplat expects them in log/logit space (before activation)
    
    def inverse_sigmoid(x):
        """Convert sigmoid-activated values to logit space."""
        return torch.log(x / (1 - x + 1e-8))
    
    # Convert existing opacities: HunyuanWorld uses sigmoid, so convert to logit
    if len(existing_opacities) > 0:
        # HunyuanWorld's opacities are already sigmoid-activated (in [0, 1])
        # Convert to logit space for gsplat
        existing_opacities = inverse_sigmoid(existing_opacities.clamp(min=1e-8, max=1-1e-8))
    
    # Convert existing scales: HunyuanWorld uses exp, so convert to log space
    if len(existing_scales) > 0:
        # HunyuanWorld's scales are already exp-activated (positive values, clamped to max 0.3)
        # Convert to log space for gsplat
        existing_scales = torch.log(existing_scales.clamp(min=1e-8))
    
    # Concatenate all parameters
    # Handle empty tensors properly
    if existing_means.numel() > 0 and new_means.numel() > 0:
        merged_means = torch.cat([existing_means, new_means], dim=0)
        merged_scales = torch.cat([existing_scales, new_scales], dim=0)
        merged_quats = torch.cat([existing_quats, new_quats], dim=0)
        merged_opacities = torch.cat([existing_opacities, new_opacities], dim=0)
        merged_sh0 = torch.cat([existing_sh0, new_sh0], dim=0)
        merged_shN = torch.cat([existing_shN, new_shN], dim=0)
    elif existing_means.numel() > 0:
        # Only existing splats
        merged_means = existing_means
        merged_scales = existing_scales
        merged_quats = existing_quats
        merged_opacities = existing_opacities
        merged_sh0 = existing_sh0
        merged_shN = existing_shN
    elif new_means.numel() > 0:
        # Only new splats
        merged_means = new_means
        merged_scales = new_scales
        merged_quats = new_quats
        merged_opacities = new_opacities
        merged_sh0 = new_sh0
        merged_shN = new_shN
    else:
        # Both empty - create empty tensors with correct shapes
        # Use device from existing_means (should always have a device even if empty)
        device = existing_means.device
        merged_means = torch.empty((0, 3), device=device, dtype=torch.float32)
        merged_scales = torch.empty((0, 3), device=device, dtype=torch.float32)
        merged_quats = torch.empty((0, 4), device=device, dtype=torch.float32)
        merged_opacities = torch.empty((0,), device=device, dtype=torch.float32)
        merged_sh0 = torch.empty((0, 1, 3), device=device, dtype=torch.float32)
        merged_shN = torch.empty((0, 8, 3), device=device, dtype=torch.float32)
    
    # Create ParameterDict
    merged_splats = nn.ParameterDict({
        "means": nn.Parameter(merged_means, requires_grad=True),
        "scales": nn.Parameter(merged_scales, requires_grad=True),
        "quats": nn.Parameter(merged_quats, requires_grad=True),
        "opacities": nn.Parameter(merged_opacities, requires_grad=True),
        "sh0": nn.Parameter(merged_sh0, requires_grad=True),
        "shN": nn.Parameter(merged_shN, requires_grad=True),
    })
    
    print(f"Merged splats: {len(existing_means)} existing + {len(new_means)} new = {len(merged_means)} total")
    
    return merged_splats


def update_3dgs_with_outpaint(
    hunyuan_predictions: Dict[str, Any],
    outpainted_image: Image.Image,
    dap_depth: np.ndarray,
    rendered_mask: np.ndarray,
    dap_validity_mask: Optional[np.ndarray] = None,
    rendered_depth: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
    depth_align_percentile: float = 60.0,
    dis_threshold: float = 0.0,
    epsilon: float = 1e-3,
    scale_rate: float = 1.0,
) -> nn.ParameterDict:
    """
    Update HunyuanWorld 3DGS with new outpainted splats.
    
    Args:
        hunyuan_predictions: HunyuanWorld predictions dict with 'splats' key
        outpainted_image: PIL Image of outpainted panorama
        dap_depth: DAP depth map [H, W] (normalized 0-1, needs scaling)
        rendered_mask: Boolean mask [H, W] - True where point cloud rendered
        dap_validity_mask: Optional validity mask from DAP [H, W]
        rendered_depth: Optional rendered depth from point cloud [H, W] (for alignment)
        device: Torch device
        depth_align_percentile: Percentile range for depth alignment
        dis_threshold: Distance threshold for valid pixels
        epsilon: Small scale for depth dimension
        scale_rate: Multiplier for scales
        
    Returns:
        Updated ParameterDict with merged splats in gsplat format
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Extract existing splats
    existing_splats = hunyuan_predictions["splats"]
    
    # Convert outpainted image to numpy
    if isinstance(outpainted_image, Image.Image):
        image_np = np.array(outpainted_image)
        if image_np.shape[2] == 4:  # RGBA
            image_np = image_np[:, :, :3]
    else:
        image_np = np.array(outpainted_image)
    
    H, W = image_np.shape[:2]
    
    # Ensure depth matches image size
    if dap_depth.shape != (H, W):
        import cv2
        dap_depth = cv2.resize(dap_depth, (W, H), interpolation=cv2.INTER_LINEAR)
    
    # Scale DAP depth from normalized [0, 1] to meters (rough estimate)
    # DAP outputs normalized depth, need to scale to reasonable range
    # Use median of non-zero values to estimate scale
    dap_depth_nonzero = dap_depth[dap_depth > 0]
    if len(dap_depth_nonzero) > 0:
        dap_median = np.median(dap_depth_nonzero)
        # Rough scaling: assume median depth is around 5-10 meters
        depth_scale_estimate = 7.0 / dap_median if dap_median > 0 else 10.0
        dap_depth_meters = dap_depth * depth_scale_estimate
    else:
        dap_depth_meters = dap_depth * 10.0  # Default scale
    
    # Align depth if rendered_depth is provided
    if rendered_depth is not None:
        # Create alignment mask: where both rendered and DAP have valid depth
        align_mask = rendered_mask & (rendered_depth > 0) & (dap_depth_meters > 0)
        if align_mask.sum() > 100:  # Need enough points for alignment
            scale = align_depth_scale(
                dap_depth_meters,
                rendered_depth,
                align_mask,
                percentile=depth_align_percentile,
            )
            dap_depth_meters = dap_depth_meters * scale
        else:
            print("Warning: Not enough overlap for depth alignment, using estimated scale")
    
    # Create mask for new content: NOT rendered AND valid DAP
    # rendered_mask: True where original point cloud rendered
    # dap_depth_valid: True where DAP produced valid depth
    dap_depth_valid = dap_depth_meters > 0.1  # Basic validity from DAP depth
    
    new_content_mask = ~rendered_mask & dap_depth_valid
    
    if dap_validity_mask is not None:
        if dap_validity_mask.shape != (H, W):
            import cv2
            dap_validity_mask = cv2.resize(
                dap_validity_mask.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST
            ) > 0.5
        new_content_mask = new_content_mask & dap_validity_mask
    
    # Convert to torch tensors
    image_torch = torch.from_numpy(image_np).float().to(device)  # [H, W, 3] uint8 -> float
    depth_torch = torch.from_numpy(dap_depth_meters).float().to(device)  # [H, W]
    mask_torch = torch.from_numpy(new_content_mask).bool().to(device)  # [H, W]
    
    # Convert masked panorama to 3DGS using convert_rgbd_equi_to_3dgs
    new_splats = convert_rgbd_equi_to_3dgs(
        rgb=image_torch,
        distance=depth_torch,
        mask=mask_torch,
        dis_threshold=dis_threshold,
        epsilon=epsilon,
        scale_rate=scale_rate,
        save_path=None,  # Don't save intermediate
    )
    
    # Merge with existing splats
    updated_splats = merge_splats(existing_splats, new_splats)
    
    return updated_splats