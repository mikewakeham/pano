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
    camera_pose_c2w: Optional[torch.Tensor] = None,
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

    if camera_pose_c2w is not None:
        centers_h = torch.cat([centers, torch.ones((centers.shape[0], 1), device=centers.device)], dim=1)
        centers = (camera_pose_c2w @ centers_h.T).T[:, :3]
        R_c2w = camera_pose_c2w[:3, :3]
        R = torch.bmm(R_c2w.unsqueeze(0).expand(R.shape[0], -1, -1), R)

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

def convert_hunyuanworld_to_gsplat(
    hunyuan_splats: Dict[str, Any],
) -> nn.ParameterDict:
    """
    Convert HunyuanWorld splats format to gsplat format.
    
    HunyuanWorld format:
        - scales/opacities: activated form (exp/sigmoid applied)
        - sh: [N, K, 3] combined SH coefficients
    
    gsplat format:
        - scales/opacities: log/logit space (before activation)
        - sh0: [N, 1, 3], shN: [N, 8, 3] split SH coefficients
    
    Args:
        hunyuan_splats: HunyuanWorld splats dict
        
    Returns:
        ParameterDict in gsplat format
    """
    # Handle batch dimension and list/tensor conversion
    def extract_tensor(key, expected_shape_suffix):
        tensor = hunyuan_splats[key]
        if isinstance(tensor, list):
            tensor = tensor[0] if len(tensor) > 0 else torch.tensor([], dtype=torch.float32)
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)
        # Remove batch dimension if present
        while tensor.ndim > len(expected_shape_suffix):
            tensor = tensor[0]
        return tensor
    
    means = extract_tensor("means", (None, 3))  # [N, 3]
    scales = extract_tensor("scales", (None, 3))  # [N, 3]
    quats = extract_tensor("quats", (None, 4))  # [N, 4]
    opacities = extract_tensor("opacities", (None,))  # [N]
    
    # Ensure opacities is 1D
    if opacities.ndim > 1:
        opacities = opacities.squeeze()
    if opacities.ndim == 0:
        opacities = opacities.unsqueeze(0)
    
    # Handle SH coefficients
    if "sh" in hunyuan_splats:
        sh = extract_tensor("sh", (None, None, 3))  # [N, K, 3]
        
        # Ensure 3D: [N, K, 3]
        if sh.ndim == 2:
            sh = sh.unsqueeze(1)  # [N, 1, 3]
        elif sh.ndim == 1:
            sh = sh.reshape(-1, 1, 3)
        
        if sh.shape[0] == 0:
            device = sh.device if sh.numel() > 0 else means.device
            sh0 = torch.zeros((0, 1, 3), device=device, dtype=sh.dtype)
            shN = torch.zeros((0, 8, 3), device=device, dtype=sh.dtype)
        elif sh.shape[1] == 1:
            sh0 = sh  # [N, 1, 3]
            shN = torch.zeros((sh0.shape[0], 8, 3), device=sh0.device, dtype=sh0.dtype)
        else:
            sh0 = sh[:, 0:1, :]  # [N, 1, 3]
            shN = sh[:, 1:, :]  # [N, K-1, 3]
            # Pad or truncate to 8 coefficients
            if shN.shape[1] < 8:
                padding = torch.zeros((shN.shape[0], 8 - shN.shape[1], 3), 
                                     device=shN.device, dtype=shN.dtype)
                shN = torch.cat([shN, padding], dim=1)
            elif shN.shape[1] > 8:
                shN = shN[:, :8, :]
    elif "colors" in hunyuan_splats:
        colors = extract_tensor("colors", (None, 3))
        sh0 = colors.unsqueeze(1)  # [N, 1, 3]
        shN = torch.zeros((sh0.shape[0], 8, 3), device=sh0.device, dtype=sh0.dtype)
    else:
        raise ValueError("hunyuan_splats must have either 'sh' or 'colors'")
    
    # Convert from activated to log/logit space
    def inverse_sigmoid(x):
        return torch.log(x / (1 - x + 1e-8))
    
    # Convert opacities: sigmoid-activated → logit space
    if opacities.numel() > 0:
        opacities = inverse_sigmoid(opacities.clamp(min=1e-8, max=1-1e-8))
    
    # Convert scales: exp-activated → log space
    if scales.numel() > 0:
        scales = torch.log(scales.clamp(min=1e-8))
    
    return nn.ParameterDict({
        "means": nn.Parameter(means, requires_grad=True),
        "scales": nn.Parameter(scales, requires_grad=True),
        "quats": nn.Parameter(quats, requires_grad=True),
        "opacities": nn.Parameter(opacities, requires_grad=True),
        "sh0": nn.Parameter(sh0, requires_grad=True),
        "shN": nn.Parameter(shN, requires_grad=True),
    })

def merge_splats(
    existing_gsplat,
    new_splats: nn.ParameterDict,
) -> nn.ParameterDict:

    # Extract tensors
    existing_means = existing_gsplat["means"].data
    existing_scales = existing_gsplat["scales"].data
    existing_quats = existing_gsplat["quats"].data
    existing_opacities = existing_gsplat["opacities"].data
    existing_sh0 = existing_gsplat["sh0"].data
    existing_shN = existing_gsplat["shN"].data
    
    # Extract new splats
    new_means = new_splats["means"].data
    new_scales = new_splats["scales"].data
    new_quats = new_splats["quats"].data
    new_opacities = new_splats["opacities"].data
    new_sh0 = new_splats["sh0"].data
    new_shN = new_splats["shN"].data
    
    # Ensure same device
    device = existing_means.device
    new_means = new_means.to(device)
    new_scales = new_scales.to(device)
    new_quats = new_quats.to(device)
    new_opacities = new_opacities.to(device)
    new_sh0 = new_sh0.to(device)
    new_shN = new_shN.to(device)
    
    # Concatenate
    if existing_means.numel() > 0 and new_means.numel() > 0:
        merged_means = torch.cat([existing_means, new_means], dim=0)
        merged_scales = torch.cat([existing_scales, new_scales], dim=0)
        merged_quats = torch.cat([existing_quats, new_quats], dim=0)
        merged_opacities = torch.cat([existing_opacities, new_opacities], dim=0)
        merged_sh0 = torch.cat([existing_sh0, new_sh0], dim=0)
        merged_shN = torch.cat([existing_shN, new_shN], dim=0)
    elif existing_means.numel() > 0:
        merged_means = existing_means
        merged_scales = existing_scales
        merged_quats = existing_quats
        merged_opacities = existing_opacities
        merged_sh0 = existing_sh0
        merged_shN = existing_shN
    elif new_means.numel() > 0:
        merged_means = new_means
        merged_scales = new_scales
        merged_quats = new_quats
        merged_opacities = new_opacities
        merged_sh0 = new_sh0
        merged_shN = new_shN
    else:
        device = existing_means.device
        merged_means = torch.empty((0, 3), device=device, dtype=torch.float32)
        merged_scales = torch.empty((0, 3), device=device, dtype=torch.float32)
        merged_quats = torch.empty((0, 4), device=device, dtype=torch.float32)
        merged_opacities = torch.empty((0,), device=device, dtype=torch.float32)
        merged_sh0 = torch.empty((0, 1, 3), device=device, dtype=torch.float32)
        merged_shN = torch.empty((0, 8, 3), device=device, dtype=torch.float32)
    
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
    base_splats,
    outpainted_image: Image.Image,
    depth: np.ndarray,
    rendered_mask: np.ndarray,
    frame_idx: int = 0,
    interpolated_camera_poses: Optional[np.ndarray] = None,
    depth_valid_mask: Optional[np.ndarray] = None,
    rendered_depth: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
    depth_align_percentile: float = 60.0,
    dis_threshold: float = 0.0,
    epsilon: float = 1e-3,
    scale_rate: float = 1.0,
) -> nn.ParameterDict:
    import cv2

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    camera_poses = (
        interpolated_camera_poses
        if interpolated_camera_poses is not None
        else hunyuan_predictions["camera_poses"][0].cpu().numpy()
    )
    cam_c2w = camera_poses[frame_idx]
    cam_c2w_torch = torch.from_numpy(cam_c2w).float().to(device)

    if isinstance(outpainted_image, Image.Image):
        image_np = np.array(outpainted_image)
        if image_np.shape[2] == 4:
            image_np = image_np[:, :, :3]
    else:
        image_np = np.array(outpainted_image)

    H, W = image_np.shape[:2]

    if depth.shape != (H, W):
        print(f"depth map dimensions {depth.shape} does not match panoramic image dimensions {image_np.shape[:2]}")
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)

    if depth_valid_mask is not None and depth_valid_mask.shape != (H, W):
        print(f"depth valid mask dimensions {depth_valid_mask.shape} does not match panoramic image dimensions {image_np.shape[:2]}")
        depth_valid_mask = cv2.resize(
            depth_valid_mask.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST
        ) > 0.5

    # Depth alignment — exclude sky from regression
    align_mask = rendered_mask & (rendered_depth > 0) & (depth > 0) if rendered_depth is not None else None
    if align_mask is not None and depth_valid_mask is not None:
        align_mask = align_mask & depth_valid_mask

    if align_mask is not None and align_mask.sum() > 100:
        scale = align_depth_scale(depth, rendered_depth, align_mask, percentile=depth_align_percentile)
        depth_scaled = depth * scale
    else:
        depth_scaled = depth.copy()

    # Set sky pixels to max valid depth
    if depth_valid_mask is not None:
        sky_far_plane = float(depth_scaled[depth_valid_mask].max())
        depth_scaled[~depth_valid_mask] = sky_far_plane
        print(f"Sky: {(~depth_valid_mask).sum()} px ({100.0 * (~depth_valid_mask).mean():.1f}%) set to far plane ({sky_far_plane:.2f})")

    new_content_mask = ~rendered_mask & (depth_scaled > 0.1)

    image_torch = torch.from_numpy(image_np).float().to(device)
    depth_torch = torch.from_numpy(depth_scaled).float().to(device)
    mask_torch  = torch.from_numpy(new_content_mask).bool().to(device)

    new_splats = convert_rgbd_equi_to_3dgs(
        rgb=image_torch,
        distance=depth_torch,
        mask=mask_torch,
        camera_pose_c2w=cam_c2w_torch,
        dis_threshold=dis_threshold,
        epsilon=epsilon,
        scale_rate=scale_rate,
        save_path=None,
    )

    return merge_splats(base_splats, new_splats)