import torch
import numpy as np
import cv2
from typing import Optional, Tuple
from scipy.spatial.transform import Rotation as R, Slerp
from gsplat.rendering import rasterization
from tqdm import tqdm
import py360convert

CUBE_FACE_DIRS = [
    ("front", np.array([0, 0, -1], np.float32), np.array([0, 1, 0], np.float32)),
    ("right", np.array([1, 0, 0], np.float32),  np.array([0, 1, 0], np.float32)),
    ("back",  np.array([0, 0, 1], np.float32),  np.array([0, 1, 0], np.float32)),
    ("left",  np.array([-1, 0, 0], np.float32), np.array([0, 1, 0], np.float32)),
    ("up",    np.array([0, -1, 0], np.float32),  np.array([0, 0, -1], np.float32)),
    ("down",  np.array([0, 1, 0], np.float32), np.array([0, 0, 1], np.float32)),
]

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
    for i in tqdm(range(num_frames)):
        cam_w2c = w2c_interp[i].unsqueeze(0)
        face_imgs = []
        for _, fwd, up in CUBE_FACE_DIRS:
            face_w2c = face_w2c_relative(fwd, up)
            face_w2c = torch.tensor(face_w2c, dtype=torch.float32, device=device).unsqueeze(0)

            R_cam, t_cam = cam_w2c[:, :3, :3], cam_w2c[:, :3, 3]
            R_face = face_w2c[:, :3, :3]

            combined = cam_w2c.clone()
            combined[:, :3, :3] = torch.bmm(R_face, R_cam)
            combined[:, :3, 3] = torch.bmm(R_face, t_cam.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                rgb, _, _ = rasterization(
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
                )
            img = (rgb[0].cpu().numpy().clip(0,1) * 255).astype(np.uint8)

            if _ in ("up", "down"):
                img = np.flip(img, axis=0)
                img = np.flip(img, axis=1)

            face_imgs.append(img)

        frames.append(np.stack(face_imgs, axis=0))  # [6,H,W,3] as [front, right, back, left, up, down]

    return np.stack(frames, axis=0)  # [N,6,H,W,3]

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