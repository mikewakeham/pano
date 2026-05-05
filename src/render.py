import torch
import numpy as np
import cv2
from typing import Optional, Tuple
from scipy.spatial.transform import Rotation as R, Slerp
from gsplat.rendering import rasterization
from tqdm import tqdm
import py360convert
from moviepy.editor import VideoFileClip
import utils3d

CUBE_FACE_DIRS = [
    ("front", np.array([0, 0, -1], np.float32), np.array([0, 1, 0], np.float32)),
    ("right", np.array([1, 0, 0], np.float32),  np.array([0, 1, 0], np.float32)),
    ("back",  np.array([0, 0, 1], np.float32),  np.array([0, 1, 0], np.float32)),
    ("left",  np.array([-1, 0, 0], np.float32), np.array([0, 1, 0], np.float32)),
    ("up",    np.array([0, -1, 0], np.float32),  np.array([0, 0, -1], np.float32)),
    ("down",  np.array([0, 1, 0], np.float32), np.array([0, 0, 1], np.float32)),
]

def spherical_uv_to_directions(uv: np.ndarray):
    theta, phi = (1 - uv[..., 0]) * (2 * np.pi), uv[..., 1] * np.pi
    directions = np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=-1)
    return directions

def get_panorama_cameras_icosahedron():
    vertices, _ = utils3d.np.create_icosahedron_mesh()
    intrinsics = utils3d.np.intrinsics_from_fov(fov_x=np.deg2rad(90), fov_y=np.deg2rad(90))
    extrinsics = utils3d.np.extrinsics_look_at([0, 0, 0], vertices, [0, 0, 1]).astype(np.float32)
    return extrinsics, [intrinsics] * len(vertices)

def render_equirectangular_torch(
    points: np.ndarray,
    colors: np.ndarray,
    H: int,
    W: int,
    camera_pose: Optional[np.ndarray] = None,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        points: [N, 3] point cloud in world coordinates (numpy array)
        colors: [N, 3] RGB colors in [0, 255] (numpy array)
        H: Height of equirectangular image
        W: Width of equirectangular image
        camera_pose: Optional [4, 4] camera pose (w2c) (numpy array)
        device: Device to use
        
    Returns:
        color_map: [H, W, 3] RGB image (uint8 numpy array on CPU)
        depth_map: [H, W] depth map (float numpy array on CPU)
        mask: [H, W] boolean mask (boolean numpy array on CPU)
    """

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    points = torch.as_tensor(points, dtype=torch.float32, device=device)
    colors = torch.as_tensor(colors, dtype=torch.float32, device=device)

    if camera_pose is not None:
        camera_pose = torch.as_tensor(camera_pose, dtype=torch.float32, device=device)
        points_h = torch.cat([points, torch.ones((points.shape[0], 1), device=device)], dim=1)
        points_cam = (camera_pose @ points_h.T).T[:, :3]
    else:
        points_cam = points

    x, y, z = points_cam.T
    r = torch.norm(points_cam, dim=1)
    
    phi = torch.arcsin(y / r)
    theta = torch.atan2(x, z)
    
    u = torch.round((theta + torch.pi) / (2 * torch.pi) * (W - 1)).long()
    v = torch.round((phi + torch.pi/2) / torch.pi * (H - 1)).long()
    
    u = torch.clamp(u, 0, W - 1)
    v = torch.clamp(v, 0, H - 1)
    
    flat_indices = v * W + u
    
    sort_idx = torch.argsort(r, descending=True)
    flat_indices_sorted = flat_indices[sort_idx]
    colors_sorted = colors[sort_idx]
    depths_sorted = r[sort_idx]
    
    color_buffer = torch.zeros((H * W, 3), dtype=torch.float32, device=device)
    depth_buffer = torch.zeros(H * W, dtype=torch.float32, device=device)
    mask_buffer = torch.zeros(H * W, dtype=torch.bool, device=device)
    
    color_buffer[flat_indices_sorted] = colors_sorted
    depth_buffer[flat_indices_sorted] = depths_sorted
    mask_buffer[flat_indices_sorted] = True
    
    color_map = color_buffer.reshape(H, W, 3)
    depth_map = depth_buffer.reshape(H, W)
    mask = mask_buffer.reshape(H, W)
    
    color_map = color_map.cpu().numpy().astype(np.uint8)
    depth_map = depth_map.cpu().numpy()
    mask = mask.cpu().numpy()
    
    del points, colors, points_cam, x, y, z, r, phi, theta, u, v
    del flat_indices, sort_idx, flat_indices_sorted, colors_sorted, depths_sorted
    del color_buffer, depth_buffer, mask_buffer
    if camera_pose is not None:
        del camera_pose
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return color_map, depth_map, mask


def pers_to_equi(
    img: np.ndarray,
    K: np.ndarray,
    H_equi: int,
    W_equi: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        img: [H, W, 3] perspective image (numpy array)
        K: [3, 3] camera intrinsics (numpy array)
        H_equi: Height of equirectangular output
        W_equi: Width of equirectangular output
        
    Returns:
        equi: [H_equi, W_equi, 3] equirectangular image (numpy array)
        mask: [H_equi, W_equi] boolean mask (boolean numpy array)
    """

    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
            
    H_p, W_p = img.shape[:2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u_grid, v_grid = np.meshgrid(np.arange(W_equi), np.arange(H_equi))
    lon = (u_grid / W_equi) * 2 * np.pi - np.pi
    lat = -((v_grid / H_equi) * np.pi - np.pi / 2)

    x = np.cos(lat) * np.sin(lon)
    y = -np.sin(lat)
    z = np.cos(lat) * np.cos(lon)

    map_x = np.full((H_equi, W_equi), -1.0, dtype=np.float32)
    map_y = np.full((H_equi, W_equi), -1.0, dtype=np.float32)

    valid_mask = z > 0
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    z_valid = z[valid_mask]

    map_x[valid_mask] = (fx * x_valid / z_valid) + cx
    map_y[valid_mask] = (fy * y_valid / z_valid) + cy

    equi = cv2.remap(
        img, 
        map_x, 
        map_y, 
        interpolation=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(0, 0, 0)
    )

    margin = 1
    mask = (map_x >= margin) & (map_x < W_p - margin) & (map_y >= margin) & (map_y < H_p - margin)
    
    mask = mask & (equi.sum(axis=2) > 0)

    return equi, mask


def c2w_to_w2c(c2w):
    is_torch = isinstance(c2w, torch.Tensor)
    if is_torch:
        R_c2w = c2w[:3, :3]
        t_c2w = c2w[:3, 3]
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ t_c2w
        w2c = torch.eye(4, dtype=c2w.dtype, device=c2w.device)
        w2c[:3, :3] = R_w2c
        w2c[:3, 3] = t_w2c
        return w2c
    else:
        R_c2w = c2w[:3, :3]
        t_c2w = c2w[:3, 3]
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ t_c2w
        w2c = np.eye(4, dtype=c2w.dtype)
        w2c[:3, :3] = R_w2c
        w2c[:3, 3] = t_w2c
        return w2c


def hunyuanworld_to_pointcloud(
    predictions: dict,
    input_images: torch.Tensor,
    num_views: Optional[int] = None,
    subsample_ratio: float = 1.0,
    confidence_threshold: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        predictions: Dict from HunyuanWorld with:
            - pts3d: [B, S, H, W, 3] 3D points in world coordinates
            - pts3d_conf: [B, S, H, W] confidence scores
        input_images: [B, N, 3, H, W] input images in [0, 1] (will be moved to CPU)
        num_views: Number of views to use (default: all)
        subsample_ratio: Ratio of points to keep (for faster processing)
        confidence_threshold: Minimum confidence to keep a point
        
    Returns:
        points: [N, 3] point cloud in world coordinates (numpy array)
        colors: [N, 3] RGB colors in [0, 255] (numpy array)
    """

    pts3d = predictions["pts3d"][0].cpu().numpy()  # [S, H, W, 3]
    pts3d_conf = predictions["pts3d_conf"][0].cpu().numpy()  # [S, H, W]
    
    if input_images.ndim == 5:
        images = input_images[0].permute(0, 2, 3, 1).cpu().numpy()  # [N, H, W, 3]
    else:
        images = input_images.permute(0, 2, 3, 1).cpu().numpy()  # [N, H, W, 3]
    
    S = pts3d.shape[0]
    if num_views is not None:
        step = max(S // num_views, 1)
        indices = np.arange(0, S, step)
    else:
        indices = np.arange(S)
    
    all_points = []
    all_colors = []
    
    for idx in indices:
        view_pts3d = pts3d[idx]  # [H, W, 3]
        view_conf = pts3d_conf[idx]  # [H, W]
        view_image = images[min(idx, images.shape[0] - 1)]  # [H, W, 3]
        
        if view_image.shape[:2] != view_pts3d.shape[:2]:
            H, W = view_pts3d.shape[:2]
            view_image = cv2.resize(
                view_image, (W, H), interpolation=cv2.INTER_LINEAR
            )
        
        valid_mask = view_conf > confidence_threshold
        
        if subsample_ratio < 1.0:
            H, W = valid_mask.shape
            step = int(1.0 / subsample_ratio) if subsample_ratio > 0 else 1
            grid_mask = np.zeros_like(valid_mask)
            grid_mask[::step, ::step] = valid_mask[::step, ::step]
            valid_mask = grid_mask
        
        points = view_pts3d[valid_mask]
        colors = view_image[valid_mask]
        
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)
        
        all_points.append(points)
        all_colors.append(colors)
    
    points = np.concatenate(all_points, axis=0)  # [N, 3]
    colors = np.concatenate(all_colors, axis=0)  # [N, 3]
    
    return points, colors

def face_w2c_relative(face_forward, face_up):
    f = face_forward / np.linalg.norm(face_forward)
    u = face_up / np.linalg.norm(face_up)
    r = np.cross(u, f); r /= np.linalg.norm(r)
    u = np.cross(f, r); u /= np.linalg.norm(u)
    R_standard = np.stack([r, u, f], axis=1)

    R_flip = np.array([[-1,0,0],[0,1,0],[0,0,-1]], np.float32)
    R_face = R_standard @ R_flip

    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R_face
    return w2c

def interpolate_w2c_poses(w2c_poses, num_frames):
    """
    Arc-length parameterized interpolation:
    - rotations: Slerp along arc-length time
    - translations: linear interpolation along arc-length time
    """
    w2c_poses = np.asarray(w2c_poses, dtype=np.float32)
    if len(w2c_poses) == num_frames:
        return w2c_poses

    positions = w2c_poses[:, :3, 3]
    dists = np.zeros(len(positions), dtype=np.float32)
    for i in range(1, len(positions)):
        dists[i] = dists[i-1] + np.linalg.norm(positions[i] - positions[i-1])

    if dists[-1] > 0:
        t_orig = dists / dists[-1]
    else:
        t_orig = np.linspace(0, 1, len(positions), dtype=np.float32)

    t_new = np.linspace(0, 1, num_frames, dtype=np.float32)

    rots = R.from_matrix(w2c_poses[:, :3, :3])
    slerp = Slerp(t_orig, rots)
    R_interp = slerp(t_new).as_matrix()

    trans = np.vstack([
        np.interp(t_new, t_orig, positions[:, 0]),
        np.interp(t_new, t_orig, positions[:, 1]),
        np.interp(t_new, t_orig, positions[:, 2]),
    ]).T

    out = np.repeat(np.eye(4, dtype=np.float32)[None], num_frames, axis=0)
    out[:, :3, :3] = R_interp
    out[:, :3, 3] = trans
    return out

def get_interpolated_camera_poses(
    video_path: str,
    camera_poses_c2w: np.ndarray,
    target_fps: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Interpolate camera poses to match target video frame rate.
    
    Args:
        video_path: Path to input video
        camera_poses_c2w: [N, 4, 4] sparse camera-to-world poses from HunyuanWorld
        target_fps: Target frames per second for rendering
        
    Returns:
        interpolated_c2w: [M, 4, 4] interpolated camera-to-world poses
        interpolated_w2c: [M, 4, 4] interpolated world-to-camera poses
        num_frames: Number of interpolated frames (M)
    """

    with VideoFileClip(video_path) as clip:
        num_frames = int(clip.duration * target_fps)
    
    camera_poses_w2c = np.stack([c2w_to_w2c(p) for p in camera_poses_c2w], axis=0)
    interpolated_w2c = interpolate_w2c_poses(camera_poses_w2c, num_frames)
    interpolated_c2w = np.stack([c2w_to_w2c(p) for p in interpolated_w2c], axis=0)
    
    return interpolated_c2w, interpolated_w2c, num_frames

def render_cubemap(
    splats: dict,
    w2c_poses: np.ndarray,
    num_frames: int,
    face_resolution: int = 512,
    device: str = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    means = splats["means"].to(device)
    quats = splats["quats"].to(device)
    scales = torch.exp(splats["scales"].to(device))
    opacities = torch.sigmoid(splats["opacities"].to(device))
    colors = torch.cat([splats["sh0"].to(device), splats["shN"].to(device)], dim=1)
    sh_degree = int(np.sqrt(colors.shape[1]) - 1)

    w2c_interp = interpolate_w2c_poses(w2c_poses, num_frames)
    w2c_interp = torch.tensor(w2c_interp, dtype=torch.float32, device=device)

    K = torch.eye(3, device=device, dtype=torch.float32)
    K[0, 0] = K[1, 1] = face_resolution / 2.0
    K[0, 2] = K[1, 2] = face_resolution / 2.0
    K = K.unsqueeze(0)

    frames = []
    alpha_frames = []
    depth_frames = []
    for i in tqdm(range(num_frames)):
        cam_w2c = w2c_interp[i].unsqueeze(0)
        face_imgs = []
        face_alphas = []
        face_depths = []
        for face_name, fwd, up in CUBE_FACE_DIRS:
            face_w2c = face_w2c_relative(fwd, up)
            face_w2c = torch.tensor(face_w2c, dtype=torch.float32, device=device).unsqueeze(0)

            R_cam, t_cam = cam_w2c[:, :3, :3], cam_w2c[:, :3, 3]
            R_face = face_w2c[:, :3, :3]

            combined = cam_w2c.clone()
            combined[:, :3, :3] = torch.bmm(R_face, R_cam)
            combined[:, :3, 3] = torch.bmm(R_face, t_cam.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                rgbd, alpha, _ = rasterization(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    viewmats=combined,
                    Ks=K,
                    width=face_resolution,
                    height=face_resolution,
                    sh_degree=sh_degree,
                    packed=True,
                    render_mode="RGB+D",
                )

            img = (rgbd[0, ..., :3].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
            depth = rgbd[0, ..., 3].cpu().numpy()
            a = alpha[0, ..., 0].cpu().numpy()

            face_imgs.append(img)
            face_alphas.append(a)
            face_depths.append(depth)

        frames.append(np.stack(face_imgs, axis=0))
        alpha_frames.append(np.stack(face_alphas, axis=0))
        depth_frames.append(np.stack(face_depths, axis=0))

    result = np.stack(frames, axis=0)  # [N,6,H,W,3]
    outputs = [result]
    outputs.append(np.stack(alpha_frames, axis=0))
    outputs.append(np.stack(depth_frames, axis=0))

    return tuple(outputs)

def cubemap_frames_to_equirectangular(
    cubemap_frames: np.ndarray,
    equi_height: int = None,
    equi_width: int = None,
):
    if cubemap_frames.ndim != 5 or cubemap_frames.shape[1] != 6:
        raise ValueError("cubemap_frames must be [N, 6, H, W, 3]")

    N, _, H, W, _ = cubemap_frames.shape
    eq_h = equi_height or H
    eq_w = equi_width or (H * 2)

    equi_frames = []
    for i in range(N):
        faces = [cubemap_frames[i, f] for f in range(6)]
        equi = py360convert.c2e(faces, eq_h, eq_w, cube_format="list")
        equi_frames.append(equi.astype(np.uint8))

    return np.stack(equi_frames, axis=0)

def cubemap_alpha_to_equirectangular_mask(
    cubemap_alpha: np.ndarray,
    equi_height: int = None,
    equi_width: int = None,
    alpha_threshold: Optional[float] = 0.5,
):
    """
    Convert per-frame cubemap alpha maps to equirectangular.

    Args:
        cubemap_alpha: [N, 6, H, W] float alpha values from 3DGS rasterization.
        equi_height: Output equirectangular height.
        equi_width: Output equirectangular width.
        alpha_threshold: If provided, pixels with alpha above this become True (binary mask).
                        If None, returns raw float alpha values (0-1).

    Returns:
        If alpha_threshold provided: [N, H_eq, W_eq] boolean mask (True/False)
        If alpha_threshold is None: [N, H_eq, W_eq] float alpha values (0-1)
    """
    if cubemap_alpha.ndim != 4 or cubemap_alpha.shape[1] != 6:
        raise ValueError("cubemap_alpha must be [N, 6, H, W]")

    N, _, H, W = cubemap_alpha.shape
    eq_h = equi_height or H
    eq_w = equi_width or (H * 2)

    results = []
    for i in range(N):
        faces = [cubemap_alpha[i, f] for f in range(6)]
        equi_alpha = py360convert.c2e(faces, eq_h, eq_w, cube_format="list")
        
        if alpha_threshold is not None:
            results.append(equi_alpha > alpha_threshold)
        else:
            results.append(equi_alpha)

    return np.stack(results, axis=0)


def cubemap_depth_to_equirectangular(
    cubemap_depth: np.ndarray,
    equi_height: int = None,
    equi_width: int = None,
):
    """
    Convert per-frame cubemap depth maps to equirectangular depth.
    """
    if cubemap_depth.ndim != 4 or cubemap_depth.shape[1] != 6:
        raise ValueError("cubemap_depth must be [N, 6, H, W]")

    N, _, H, W = cubemap_depth.shape
    eq_h = equi_height or H
    eq_w = equi_width or (H * 2)

    depth_frames = []
    for i in range(N):
        faces = [cubemap_depth[i, f] for f in range(6)]
        equi_depth = py360convert.c2e(faces, eq_h, eq_w, cube_format="list")
        depth_frames.append(equi_depth.astype(np.float32))

    return np.stack(depth_frames, axis=0)


def render_3dgs_alpha_with_radius_filter(
    splats: dict,
    w2c_pose: np.ndarray,
    equi_height: int,
    equi_width: int,
    face_resolution: int = 512,
    device: str = None,
    max_gaussian_radius_px: float = 64.0,
    radius_percentile: float = 99.0,
    scale_percentile: float = 99.5,
) -> np.ndarray:
    """
    Render an equirectangular alpha map for masking, excluding gaussians whose
    projected footprint is too large on a cubemap face.

    The filtering is face-wise:
    - probe rasterization measures projected radii for visible gaussians
    - gaussians with outlier 2D radii are removed from the mask render
    - a world-scale percentile guard removes persistent huge splats
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    means = splats["means"].to(device)
    quats = splats["quats"].to(device)
    scales = torch.exp(splats["scales"].to(device))
    opacities = torch.sigmoid(splats["opacities"].to(device))
    colors = torch.cat([splats["sh0"].to(device), splats["shN"].to(device)], dim=1)
    sh_degree = int(np.sqrt(colors.shape[1]) - 1)

    cam_w2c = torch.tensor(w2c_pose, dtype=torch.float32, device=device).unsqueeze(0)

    K = torch.eye(3, device=device, dtype=torch.float32)
    K[0, 0] = K[1, 1] = face_resolution / 2.0
    K[0, 2] = K[1, 2] = face_resolution / 2.0
    K = K.unsqueeze(0)

    scale_metric = torch.amax(scales, dim=1).to(torch.float32)
    scale_cutoff = torch.quantile(
        scale_metric, torch.tensor(scale_percentile / 100.0, device=device)
    )

    face_alphas = []
    for face_name, fwd, up in CUBE_FACE_DIRS:
        face_w2c = face_w2c_relative(fwd, up)
        face_w2c = torch.tensor(face_w2c, dtype=torch.float32, device=device).unsqueeze(0)

        R_cam, t_cam = cam_w2c[:, :3, :3], cam_w2c[:, :3, 3]
        R_face = face_w2c[:, :3, :3]

        combined = cam_w2c.clone()
        combined[:, :3, :3] = torch.bmm(R_face, R_cam)
        combined[:, :3, 3] = torch.bmm(R_face, t_cam.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            _, _, info = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=combined,
                Ks=K,
                width=face_resolution,
                height=face_resolution,
                sh_degree=sh_degree,
                packed=False,
            )

        radii = info["radii"]
        if radii.ndim == 3:
            if radii.shape[0] != 1:
                raise ValueError(f"Expected single-camera radii, got shape {tuple(radii.shape)}")
            radii = radii[0]
        if radii.ndim != 2 or radii.shape[-1] != 2:
            raise ValueError(f"Unexpected gsplat radii shape {tuple(radii.shape)}")

        visible = (radii > 0).all(dim=-1)
        footprint = radii.to(torch.float32).amax(dim=-1)

        if torch.any(visible):
            radius_cutoff = torch.quantile(
                footprint[visible], torch.tensor(radius_percentile / 100.0, device=device)
            )
            radius_cutoff = torch.minimum(
                radius_cutoff,
                torch.tensor(max_gaussian_radius_px, dtype=footprint.dtype, device=device),
            )
            keep = visible & (footprint <= radius_cutoff) & (scale_metric <= scale_cutoff)
        else:
            keep = torch.ones_like(scale_metric, dtype=torch.bool)

        if not torch.any(keep):
            keep = visible & (scale_metric <= scale_cutoff)
        if not torch.any(keep):
            keep = visible
        if not torch.any(keep):
            keep = torch.ones_like(scale_metric, dtype=torch.bool)

        with torch.no_grad():
            _, alpha, _ = rasterization(
                means=means[keep],
                quats=quats[keep],
                scales=scales[keep],
                opacities=opacities[keep],
                colors=colors[keep],
                viewmats=combined,
                Ks=K,
                width=face_resolution,
                height=face_resolution,
                sh_degree=sh_degree,
                packed=False,
            )

        a = alpha[0, ..., 0].cpu().numpy()
        if face_name in ("up", "down"):
            a = np.flip(a, axis=0)
            a = np.flip(a, axis=1)
        face_alphas.append(a)

    cubemap_alpha = np.expand_dims(np.stack(face_alphas, axis=0), axis=0)
    equi_alpha = cubemap_alpha_to_equirectangular_mask(
        cubemap_alpha=cubemap_alpha,
        equi_height=equi_height,
        equi_width=equi_width,
        alpha_threshold=None,
    )
    return equi_alpha[0]


def render_3dgs_to_equirectangular(
    splats: dict,
    w2c_pose: np.ndarray,
    equi_height: int,
    equi_width: int,
    face_resolution: int = 512,
    device: str = None,
    alpha_threshold: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render 3DGS to a single equirectangular image at the given camera pose.

    Args:
        splats: 3DGS splats dict (means, quats, scales, opacities, sh0, shN)
        w2c_pose: [4, 4] camera pose (world-to-camera)
        equi_height, equi_width: Output equirectangular dimensions
        face_resolution: Cubemap face resolution
        device: Torch device

    Returns:
        RGB image [H, W, 3] and either raw alpha [H, W] or a boolean mask [H, W]
        depending on `alpha_threshold`.
    """
    w2c_poses = np.expand_dims(w2c_pose, axis=0)
    cubemap_frames, cubemap_alpha, _ = render_cubemap(
        splats=splats,
        w2c_poses=w2c_poses,
        num_frames=1,
        face_resolution=face_resolution,
        device=device,
    )
    
    equi_rgb = cubemap_frames_to_equirectangular(
        cubemap_frames=cubemap_frames,
        equi_height=equi_height,
        equi_width=equi_width,
    )
    
    equi_alpha = cubemap_alpha_to_equirectangular_mask(
        cubemap_alpha=cubemap_alpha,
        equi_height=equi_height,
        equi_width=equi_width,
        alpha_threshold=alpha_threshold,
    )
    
    return equi_rgb[0], equi_alpha[0]  # [H, W, 3], [H, W]


def render_icosahedron(
    splats: dict,
    w2c_poses: np.ndarray,
    num_frames: int,
    face_resolution: int = 512,
    batch_size: int = 6,
    device: str = None,
):
    """
    Render 3DGS using icosahedron views per frame with batched view processing.
    
    Args:
        splats: 3DGS splats dict
        w2c_poses: [N, 4, 4] world-to-camera poses
        num_frames: Number of frames to render
        face_resolution: Resolution for each perspective view
        batch_size: Number of views to render at once per frame (default: 6)
        device: Torch device
        
    Returns:
        ico_frames: [N, num_views, H, W, 3] RGB images
        ico_alpha: [N, num_views, H, W] alpha values
        ico_depth: [N, num_views, H, W] depth values
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    means = splats["means"].to(device)
    quats = splats["quats"].to(device)
    scales = torch.exp(splats["scales"].to(device))
    opacities = torch.sigmoid(splats["opacities"].to(device))
    colors = torch.cat([splats["sh0"].to(device), splats["shN"].to(device)], dim=1)
    sh_degree = int(np.sqrt(colors.shape[1]) - 1)

    # Get icosahedron camera directions
    extrinsics_ico, intrinsics_ico = get_panorama_cameras_icosahedron()
    num_views = len(extrinsics_ico)
    
    # Interpolate camera poses
    w2c_interp = interpolate_w2c_poses(w2c_poses, num_frames)
    w2c_interp = torch.tensor(w2c_interp, dtype=torch.float32, device=device)

    # Setup intrinsics for perspective rendering
    K = torch.eye(3, device=device, dtype=torch.float32)
    K[0, 0] = K[1, 1] = face_resolution / 2.0
    K[0, 2] = K[1, 2] = face_resolution / 2.0
    K = K.unsqueeze(0)

    # Pre-allocate output arrays
    all_frames = np.zeros((num_frames, num_views, face_resolution, face_resolution, 3), dtype=np.uint8)
    all_alphas = np.zeros((num_frames, num_views, face_resolution, face_resolution), dtype=np.float32)
    all_depths = np.zeros((num_frames, num_views, face_resolution, face_resolution), dtype=np.float32)
    
    # Progress bar over frames
    pbar = tqdm(total=num_frames, desc="Rendering")
    
    # Render each frame
    for frame_idx in range(num_frames):
        cam_w2c = w2c_interp[frame_idx].unsqueeze(0)  # [1, 4, 4]
        
        # Batch process views for this frame
        for batch_start in range(0, num_views, batch_size):
            batch_end = min(batch_start + batch_size, num_views)
            batch_views = batch_end - batch_start
            
            # Get view extrinsics for this batch
            view_extrinsics_batch = [extrinsics_ico[i] for i in range(batch_start, batch_end)]
            view_w2c_batch = torch.tensor(
                np.stack(view_extrinsics_batch), 
                dtype=torch.float32, 
                device=device
            )  # [B_views, 4, 4]
            
            # Combine camera pose with each view direction
            R_cam = cam_w2c[:, :3, :3]  # [1, 3, 3]
            t_cam = cam_w2c[:, :3, 3]   # [1, 3]
            R_views = view_w2c_batch[:, :3, :3]  # [B_views, 3, 3]
            
            # Broadcast camera to match batch
            combined = cam_w2c.expand(batch_views, -1, -1).clone()  # [B_views, 4, 4]
            combined[:, :3, :3] = torch.bmm(R_views, R_cam.expand(batch_views, -1, -1))
            combined[:, :3, 3] = torch.bmm(
                R_views, 
                t_cam.unsqueeze(-1).expand(batch_views, -1, -1)
            ).squeeze(-1)
            
            # Replicate K for batch
            K_batch = K.expand(batch_views, -1, -1)
            
            with torch.no_grad():
                rgbd, alpha, _ = rasterization(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    viewmats=combined,
                    Ks=K_batch,
                    width=face_resolution,
                    height=face_resolution,
                    sh_degree=sh_degree,
                    packed=True,
                    render_mode="RGB+D",
                )

            # Extract and store results
            imgs = (rgbd[..., :3].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
            depths = rgbd[..., 3].cpu().numpy()
            alphas = alpha[..., 0].cpu().numpy()
            
            all_frames[frame_idx, batch_start:batch_end] = imgs
            all_alphas[frame_idx, batch_start:batch_end] = alphas
            all_depths[frame_idx, batch_start:batch_end] = depths
        
        # Update progress bar once per frame (after all batches for this frame)
        pbar.update(1)
    
    pbar.close()
    
    return all_frames, all_alphas, all_depths


def icosahedron_views_to_equirectangular(
    ico_views: np.ndarray,
    equi_height: int = None,
    equi_width: int = None,
    blend_mode: str = "weighted",
):
    """
    Convert icosahedron views to equirectangular - MoGe-compatible version.
    """
    if ico_views.ndim != 5:
        raise ValueError(f"ico_views must be [N, num_views, H, W, C], got shape {ico_views.shape}")

    N, num_views, H, W, C = ico_views.shape
    eq_h = equi_height or H
    eq_w = equi_width or (H * 2)
    
    # Get icosahedron cameras
    extrinsics_ico, intrinsics_ico = get_panorama_cameras_icosahedron()
    if len(extrinsics_ico) != num_views:
        raise ValueError(f"Expected {len(extrinsics_ico)} views but got {num_views}")
    
    # Pre-compute mappings
    print("Pre-computing projection mappings...")
    view_mappings = []
    
    for view_idx in tqdm(range(num_views), desc="Pre-computing", leave=False):
        # Create UV map for equirectangular output
        uv_equi = utils3d.np.uv_map(eq_h, eq_w)  # [H, W, 2]
        
        # Convert equi UV to 3D directions (in world space)
        spherical_directions = spherical_uv_to_directions(uv_equi)  # [H, W, 3]
        
        # Project these world directions into this camera's view
        # This gives us where each equi pixel maps to in the perspective view
        projected_uv, projected_depth = utils3d.np.project_cv(
            spherical_directions,
            extrinsics=extrinsics_ico[view_idx],
            intrinsics=intrinsics_ico[view_idx]
        )
        
        # Check validity
        projection_valid = (
            (projected_depth > 0) & 
            (projected_uv[..., 0] > 0) & (projected_uv[..., 0] < 1) &
            (projected_uv[..., 1] > 0) & (projected_uv[..., 1] < 1)
        )
        
        # Convert UV [0,1] to pixel coordinates
        projected_pixels = utils3d.np.uv_to_pixel(
            np.clip(projected_uv, 0, 1), 
            (H, W)
        ).astype(np.float32)
        
        # Compute weights (angle-based)
        view_forward = -extrinsics_ico[view_idx][:3, 2]
        cos_angle = np.sum(spherical_directions * view_forward, axis=-1)
        cos_angle = np.clip(cos_angle, 0, 1)
        weight = (cos_angle ** 2) * projection_valid.astype(np.float32)
        
        view_mappings.append({
            'pixels': projected_pixels,
            'valid': projection_valid,
            'weight': weight[..., None]  # Add channel dim
        })
    
    # Process frames
    equi_frames = []
    
    for frame_idx in tqdm(range(N), desc="Converting to equirectangular"):
        if blend_mode == "weighted":
            output = np.zeros((eq_h, eq_w, C), dtype=np.float32)
            weight_sum = np.zeros((eq_h, eq_w, 1), dtype=np.float32)
            
            for view_idx in range(num_views):
                mapping = view_mappings[view_idx]
                
                if not np.any(mapping['valid']):
                    continue
                
                view_img = ico_views[frame_idx, view_idx]
                
                # Remap using pre-computed mappings
                if C == 3:
                    sampled = cv2.remap(
                        view_img,
                        mapping['pixels'][..., 0],
                        mapping['pixels'][..., 1],
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0)
                    )
                else:
                    sampled = cv2.remap(
                        view_img,
                        mapping['pixels'][..., 0],
                        mapping['pixels'][..., 1],
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0
                    )
                    if sampled.ndim == 2:
                        sampled = sampled[..., None]
                
                output += sampled * mapping['weight']
                weight_sum += mapping['weight']
            
            # Normalize
            valid = weight_sum > 1e-6
            output[valid[..., 0]] /= weight_sum[valid[..., 0]]
            
        else:  # nearest - use same pre-computed mappings
            output = np.zeros((eq_h, eq_w, C), dtype=np.float32)
            
            # For each pixel, find view with best angle
            best_weights = np.zeros((eq_h, eq_w))
            best_view = np.zeros((eq_h, eq_w), dtype=np.int32)
            
            for view_idx in range(num_views):
                mapping = view_mappings[view_idx]
                weight = mapping['weight'][..., 0]
                mask = weight > best_weights
                best_weights[mask] = weight[mask]
                best_view[mask] = view_idx
            
            # Sample from best view for each pixel
            for view_idx in range(num_views):
                mask = best_view == view_idx
                if not np.any(mask):
                    continue
                
                mapping = view_mappings[view_idx]
                view_img = ico_views[frame_idx, view_idx]
                
                if C == 3:
                    sampled = cv2.remap(
                        view_img,
                        mapping['pixels'][..., 0],
                        mapping['pixels'][..., 1],
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0)
                    )
                else:
                    sampled = cv2.remap(
                        view_img,
                        mapping['pixels'][..., 0],
                        mapping['pixels'][..., 1],
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0
                    )
                    if sampled.ndim == 2:
                        sampled = sampled[..., None]
                
                output[mask] = sampled[mask]
        
        if C == 3:
            output = output.astype(np.uint8)
        equi_frames.append(output.squeeze())
    
    result = np.stack(equi_frames, axis=0)
    if C == 1 and result.ndim == 3:
        result = result[..., None]
    return result

def prune_reference_view_floaters(
    base_splats: dict,
    w2c_pose: np.ndarray,
    equi_height: int,
    equi_width: int,
    face_resolution: int = 512,
    device: str = None,
    radius_percentile: float = 99.8,
    max_candidates: int = 128,
    support_alpha_threshold: float = 0.10,
    candidate_alpha_threshold: float = 0.08,
    outside_support_ratio_threshold: float = 0.35,
    solo_ratio_threshold: float = 0.50,
    overlap_eps: float = 0.05,
) -> dict:
    """
    Prune base gsplat splats using one reference view.

    A splat is removed only if:
    - it is among the largest projected splats in the chosen view
    - much of its projected area lies outside the main support region
    - much of its projected area is effectively solo
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    means = base_splats["means"].to(device)
    quats = base_splats["quats"].to(device)
    scales = torch.exp(base_splats["scales"].to(device))
    opacities = torch.sigmoid(base_splats["opacities"].to(device))
    colors = torch.cat([base_splats["sh0"].to(device), base_splats["shN"].to(device)], dim=1)
    sh_degree = int(np.sqrt(colors.shape[1]) - 1)

    cam_w2c = torch.tensor(w2c_pose, dtype=torch.float32, device=device).unsqueeze(0)

    K = torch.eye(3, device=device, dtype=torch.float32)
    K[0, 0] = K[1, 1] = face_resolution / 2.0
    K[0, 2] = K[1, 2] = face_resolution / 2.0
    K = K.unsqueeze(0)

    def _render_equi_alpha(keep_mask: torch.Tensor) -> np.ndarray:
        face_alphas = []
        for face_name, fwd, up in CUBE_FACE_DIRS:
            face_w2c = face_w2c_relative(fwd, up)
            face_w2c = torch.tensor(face_w2c, dtype=torch.float32, device=device).unsqueeze(0)

            R_cam, t_cam = cam_w2c[:, :3, :3], cam_w2c[:, :3, 3]
            R_face = face_w2c[:, :3, :3]

            combined = cam_w2c.clone()
            combined[:, :3, :3] = torch.bmm(R_face, R_cam)
            combined[:, :3, 3] = torch.bmm(R_face, t_cam.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                _, alpha, _ = rasterization(
                    means=means[keep_mask],
                    quats=quats[keep_mask],
                    scales=scales[keep_mask],
                    opacities=opacities[keep_mask],
                    colors=colors[keep_mask],
                    viewmats=combined,
                    Ks=K,
                    width=face_resolution,
                    height=face_resolution,
                    sh_degree=sh_degree,
                    packed=False,
                )

            a = alpha[0, ..., 0].cpu().numpy()
            if face_name in ("up", "down"):
                a = np.flip(a, axis=0)
                a = np.flip(a, axis=1)
            face_alphas.append(a)

        cubemap_alpha = np.expand_dims(np.stack(face_alphas, axis=0), axis=0)
        equi_alpha = cubemap_alpha_to_equirectangular_mask(
            cubemap_alpha=cubemap_alpha,
            equi_height=equi_height,
            equi_width=equi_width,
            alpha_threshold=None,
        )
        return equi_alpha[0]

    full_keep = torch.ones(means.shape[0], dtype=torch.bool, device=device)
    full_alpha = _render_equi_alpha(full_keep)
    support_mask = full_alpha >= support_alpha_threshold

    max_footprint = torch.zeros(means.shape[0], dtype=torch.float32, device=device)

    for _, fwd, up in CUBE_FACE_DIRS:
        face_w2c = face_w2c_relative(fwd, up)
        face_w2c = torch.tensor(face_w2c, dtype=torch.float32, device=device).unsqueeze(0)

        R_cam, t_cam = cam_w2c[:, :3, :3], cam_w2c[:, :3, 3]
        R_face = face_w2c[:, :3, :3]

        combined = cam_w2c.clone()
        combined[:, :3, :3] = torch.bmm(R_face, R_cam)
        combined[:, :3, 3] = torch.bmm(R_face, t_cam.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            _, _, info = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=combined,
                Ks=K,
                width=face_resolution,
                height=face_resolution,
                sh_degree=sh_degree,
                packed=False,
            )

        radii = info["radii"]
        if radii.ndim == 3:
            radii = radii[0]

        visible = (radii > 0).all(dim=-1)
        footprint = radii.to(torch.float32).amax(dim=-1)
        max_footprint = torch.maximum(
            max_footprint,
            torch.where(visible, footprint, torch.zeros_like(footprint)),
        )

    visible_any = max_footprint > 0
    if not torch.any(visible_any):
        return base_splats

    cutoff = torch.quantile(
        max_footprint[visible_any],
        torch.tensor(radius_percentile / 100.0, device=device),
    )

    candidate_idx = torch.nonzero(max_footprint >= cutoff, as_tuple=False).squeeze(1)
    if candidate_idx.numel() == 0:
        return base_splats

    if candidate_idx.numel() > max_candidates:
        candidate_scores = max_footprint[candidate_idx]
        topk = torch.topk(candidate_scores, k=max_candidates).indices
        candidate_idx = candidate_idx[topk]

    bad = torch.zeros(means.shape[0], dtype=torch.bool, device=device)

    for idx in candidate_idx.tolist():
        keep_mask = torch.zeros(means.shape[0], dtype=torch.bool, device=device)
        keep_mask[idx] = True

        cand_alpha = _render_equi_alpha(keep_mask)
        cand_mask = cand_alpha >= candidate_alpha_threshold
        cand_area = int(cand_mask.sum())
        if cand_area == 0:
            continue

        outside_support_ratio = float((cand_mask & (~support_mask)).sum()) / float(cand_area)
        solo_mask = cand_mask & ((full_alpha - cand_alpha) <= overlap_eps)
        solo_ratio = float(solo_mask.sum()) / float(cand_area)

        if outside_support_ratio >= outside_support_ratio_threshold and solo_ratio >= solo_ratio_threshold:
            bad[idx] = True

    keep = ~bad
    print(f"Reference-view floater pruning removed {(~keep).sum().item()} / {keep.numel()} gaussians")

    return {
        k: v[keep.to(v.device)] if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == keep.numel() else v
        for k, v in base_splats.items()
    }


def prune_hunyuan_reference_view_floaters(
    predictions: dict,
    frame_idx: int,
    camera_poses_c2w: Optional[np.ndarray] = None,
    face_resolution: int = 512,
    equi_height: int = 1024,
    equi_width: int = 2048,
    device: str = None,
    radius_percentile: float = 99.8,
    max_candidates: int = 256,
    support_alpha_threshold: float = 0.10,
    candidate_alpha_threshold: float = 0.08,
    outside_support_ratio_threshold: float = 0.35,
    solo_ratio_threshold: float = 0.50,
    overlap_eps: float = 0.05,
    support_kernel_size: int = 9,
) -> dict:
    """
    Prune HunyuanWorld floaters using a single reference view.

    A candidate gaussian is removed only if:
    - it is among the largest projected splats in the chosen frame
    - a substantial part of its projected area lies outside the main scene support
    - much of its projected area is effectively solo (little support from other splats)

    Returns a pruned gsplat-format splat dict.
    """
    from update_3dgs import convert_hunyuanworld_to_gsplat

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    gsplat_splats = convert_hunyuanworld_to_gsplat(predictions["splats"])

    if camera_poses_c2w is None:
        camera_poses_c2w = predictions["camera_poses"][0].cpu().numpy()

    cam_c2w = camera_poses_c2w[frame_idx]
    w2c_pose = c2w_to_w2c(cam_c2w)

    means = gsplat_splats["means"].to(device)
    quats = gsplat_splats["quats"].to(device)
    scales = torch.exp(gsplat_splats["scales"].to(device))
    opacities = torch.sigmoid(gsplat_splats["opacities"].to(device))
    colors = torch.cat([gsplat_splats["sh0"].to(device), gsplat_splats["shN"].to(device)], dim=1)
    sh_degree = int(np.sqrt(colors.shape[1]) - 1)

    cam_w2c = torch.tensor(w2c_pose, dtype=torch.float32, device=device).unsqueeze(0)

    K = torch.eye(3, device=device, dtype=torch.float32)
    K[0, 0] = K[1, 1] = face_resolution / 2.0
    K[0, 2] = K[1, 2] = face_resolution / 2.0
    K = K.unsqueeze(0)

    def _render_alpha_from_mask(keep_mask: torch.Tensor) -> np.ndarray:
        face_alphas = []
        for face_name, fwd, up in CUBE_FACE_DIRS:
            face_w2c = face_w2c_relative(fwd, up)
            face_w2c = torch.tensor(face_w2c, dtype=torch.float32, device=device).unsqueeze(0)

            R_cam, t_cam = cam_w2c[:, :3, :3], cam_w2c[:, :3, 3]
            R_face = face_w2c[:, :3, :3]

            combined = cam_w2c.clone()
            combined[:, :3, :3] = torch.bmm(R_face, R_cam)
            combined[:, :3, 3] = torch.bmm(R_face, t_cam.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                _, alpha, _ = rasterization(
                    means=means[keep_mask],
                    quats=quats[keep_mask],
                    scales=scales[keep_mask],
                    opacities=opacities[keep_mask],
                    colors=colors[keep_mask],
                    viewmats=combined,
                    Ks=K,
                    width=face_resolution,
                    height=face_resolution,
                    sh_degree=sh_degree,
                    packed=False,
                )

            a = alpha[0, ..., 0].cpu().numpy()
            if face_name in ("up", "down"):
                a = np.flip(a, axis=0)
                a = np.flip(a, axis=1)
            face_alphas.append(a)

        cubemap_alpha = np.expand_dims(np.stack(face_alphas, axis=0), axis=0)
        equi_alpha = cubemap_alpha_to_equirectangular_mask(
            cubemap_alpha=cubemap_alpha,
            equi_height=equi_height,
            equi_width=equi_width,
            alpha_threshold=None,
        )
        return equi_alpha[0]

    # Full support map in panorama space.
    full_keep = torch.ones(means.shape[0], dtype=torch.bool, device=device)
    full_alpha = _render_alpha_from_mask(full_keep)

    support_mask = (full_alpha >= support_alpha_threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (support_kernel_size, support_kernel_size)
    )
    support_mask = cv2.morphologyEx(support_mask, cv2.MORPH_CLOSE, kernel)
    support_mask = cv2.morphologyEx(support_mask, cv2.MORPH_OPEN, kernel).astype(bool)

    # Find large projected candidates from the chosen view.
    max_footprint = torch.zeros(means.shape[0], dtype=torch.float32, device=device)

    for _, fwd, up in CUBE_FACE_DIRS:
        face_w2c = face_w2c_relative(fwd, up)
        face_w2c = torch.tensor(face_w2c, dtype=torch.float32, device=device).unsqueeze(0)

        R_cam, t_cam = cam_w2c[:, :3, :3], cam_w2c[:, :3, 3]
        R_face = face_w2c[:, :3, :3]

        combined = cam_w2c.clone()
        combined[:, :3, :3] = torch.bmm(R_face, R_cam)
        combined[:, :3, 3] = torch.bmm(R_face, t_cam.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            _, _, info = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=combined,
                Ks=K,
                width=face_resolution,
                height=face_resolution,
                sh_degree=sh_degree,
                packed=False,
            )

        radii = info["radii"]
        if radii.ndim == 3:
            radii = radii[0]
        visible = (radii > 0).all(dim=-1)
        footprint = radii.to(torch.float32).amax(dim=-1)
        max_footprint = torch.maximum(max_footprint, torch.where(visible, footprint, torch.zeros_like(footprint)))

    visible_any = max_footprint > 0
    if not torch.any(visible_any):
        return gsplat_splats

    cutoff = torch.quantile(
        max_footprint[visible_any],
        torch.tensor(radius_percentile / 100.0, device=device),
    )
    candidate_idx = torch.nonzero(max_footprint >= cutoff, as_tuple=False).squeeze(1)

    if candidate_idx.numel() == 0:
        return gsplat_splats

    if candidate_idx.numel() > max_candidates:
        candidate_scores = max_footprint[candidate_idx]
        topk = torch.topk(candidate_scores, k=max_candidates).indices
        candidate_idx = candidate_idx[topk]

    bad = torch.zeros(means.shape[0], dtype=torch.bool, device=device)

    for idx in candidate_idx.tolist():
        keep_mask = torch.zeros(means.shape[0], dtype=torch.bool, device=device)
        keep_mask[idx] = True

        cand_alpha = _render_alpha_from_mask(keep_mask)
        cand_mask = cand_alpha >= candidate_alpha_threshold
        cand_area = int(cand_mask.sum())
        if cand_area == 0:
            continue

        outside_support_ratio = float((cand_mask & (~support_mask)).sum()) / float(cand_area)

        # Approximate "solo" as pixels where this gaussian explains almost all observed alpha.
        solo_mask = cand_mask & ((full_alpha - cand_alpha) <= overlap_eps)
        solo_ratio = float(solo_mask.sum()) / float(cand_area)

        if outside_support_ratio >= outside_support_ratio_threshold and solo_ratio >= solo_ratio_threshold:
            bad[idx] = True

    keep = ~bad
    print(f"Reference-view floater pruning removed {(~keep).sum().item()} / {keep.numel()} gaussians")

    return {
        k: v[keep] if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == keep.numel() else v
        for k, v in gsplat_splats.items()
    }


def render_panoramic_video(
    splats: dict,
    w2c_poses: np.ndarray,
    num_frames: int,
    render_type: str = "cubemap",
    face_resolution: int = 512,
    equi_height: int = 1024,
    equi_width: int = 2048,
    alpha_threshold: float = 0.8,
    blend_mode: str = "weighted",
    device: str = None,
):
    """
    Unified function to render 3DGS to equirectangular video using either cubemap or icosahedron.
    
    Args:
        splats: 3DGS splats dict
        w2c_poses: [N, 4, 4] world-to-camera poses
        num_frames: Number of frames to render
        render_type: "cubemap" (6 faces) or "icosahedron"
        face_resolution: Resolution for each face/view
        equi_height: Output equirectangular height
        equi_width: Output equirectangular width
        alpha_threshold: Threshold for binary mask (if None, returns raw alpha)
        blend_mode: For icosahedron: "weighted" or "nearest" (ignored for cubemap)
        device: Torch device
        
    Returns:
        equirectangular_frames: [N, H, W, 3] RGB frames
        equirectangular_masks: [N, H, W] boolean masks (or float alpha if alpha_threshold=None)
        equirectangular_depth: [N, H, W] depth maps
    """
    if render_type == "cubemap":
        print(f"Rendering {num_frames} frames using cubemap (6 faces)...")
        cubemap_frames, cubemap_alpha, cubemap_depth = render_cubemap(
            splats=splats,
            w2c_poses=w2c_poses,
            num_frames=num_frames,
            face_resolution=face_resolution,
            device=device,
        )
        
        equirectangular_frames = cubemap_frames_to_equirectangular(
            cubemap_frames=cubemap_frames,
            equi_height=equi_height,
            equi_width=equi_width,
        )
        
        equirectangular_masks = cubemap_alpha_to_equirectangular_mask(
            cubemap_alpha=cubemap_alpha,
            equi_height=equi_height,
            equi_width=equi_width,
            alpha_threshold=alpha_threshold,
        )
        
        equirectangular_depth = cubemap_depth_to_equirectangular(
            cubemap_depth=cubemap_depth,
            equi_height=equi_height,
            equi_width=equi_width,
        )
        
    elif render_type == "icosahedron":
        print(f"Rendering {num_frames} frames using icosahedron...")
        ico_frames, ico_alpha, ico_depth = render_icosahedron(
            splats=splats,
            w2c_poses=w2c_poses,
            num_frames=num_frames,
            face_resolution=face_resolution,
            device=device,
        )
        
        equirectangular_frames = icosahedron_views_to_equirectangular(
            ico_views=ico_frames,
            equi_height=equi_height,
            equi_width=equi_width,
            blend_mode=blend_mode,
        )
        
        # Convert alpha
        ico_alpha_expanded = ico_alpha[..., None]  # [N, 20, H, W, 1]
        equi_alpha = icosahedron_views_to_equirectangular(
            ico_views=ico_alpha_expanded,
            equi_height=equi_height,
            equi_width=equi_width,
            blend_mode=blend_mode,
        )
        equi_alpha = equi_alpha.squeeze(-1)  # [N, H, W]
        
        if alpha_threshold is not None:
            equirectangular_masks = equi_alpha > alpha_threshold
        else:
            equirectangular_masks = equi_alpha
        
        # Convert depth
        ico_depth_expanded = ico_depth[..., None]  # [N, 20, H, W, 1]
        equirectangular_depth = icosahedron_views_to_equirectangular(
            ico_views=ico_depth_expanded,
            equi_height=equi_height,
            equi_width=equi_width,
            blend_mode=blend_mode,
        )
        equirectangular_depth = equirectangular_depth.squeeze(-1)  # [N, H, W]
        
    else:
        raise ValueError(f"Invalid render_type: {render_type}. Must be 'cubemap' or 'icosahedron'")
    
    return equirectangular_frames, equirectangular_masks, equirectangular_depth