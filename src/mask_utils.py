import numpy as np
import cv2
from scipy.interpolate import griddata
from typing import Tuple


def clean_mask(
    rendered_mask: np.ndarray,
    kernel_size: int = 3,
    min_region_size: int = 100,
) -> np.ndarray:
    """
    1. Morphological closing to fill small gaps
    2. Connected components analysis
    3. Remove small regions
    """

    rendered_mask = rendered_mask.astype(np.uint8)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*kernel_size+1, 2*kernel_size+1))
    closed = cv2.morphologyEx(rendered_mask, cv2.MORPH_CLOSE, kernel)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    
    refined_mask = np.zeros_like(closed)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_region_size:
            refined_mask[labels == lbl] = 1
    
    return refined_mask.astype(bool)


def fill_holes_with_interpolation(
    combined_rgb: np.ndarray,
    rendered_mask: np.ndarray,
    refined_mask: np.ndarray,
) -> np.ndarray:
    """
    Fill holes in masked region using color interpolation
    
    Args:
        combined_rgb: [H, W, 3] RGB image
        rendered_mask: [H, W] original boolean mask
        refined_mask: [H, W] refined boolean mask (may have filled holes)
        
    Returns:
        filled_rgb: [H, W, 3] RGB image with filled holes
    """

    hole_mask = np.logical_and(refined_mask == 1, rendered_mask == 0)
    
    ys, xs = np.nonzero(rendered_mask == 1)
    if len(ys) == 0:
        return combined_rgb.copy()
    known_colors = combined_rgb[ys, xs]
    
    yy, xx = np.nonzero(hole_mask)
    coords_known = np.stack([ys, xs], axis=-1)
    coords_fill = np.stack([yy, xx], axis=-1)
    
    filled_rgb = combined_rgb.copy()
    for c in range(3):
        channel_values = known_colors[:, c]
        filled_rgb[yy, xx, c] = griddata(
            coords_known, channel_values, coords_fill, method='linear', fill_value=0
        )
    
    nan_mask = np.isnan(filled_rgb)
    for c in range(3):
        filled_rgb[nan_mask[..., c], c] = combined_rgb[nan_mask[..., c], c]
    
    return filled_rgb


def fill_holes_telea(rgb, alpha, source_threshold=0.9, radius=15):
    rgb_uint8 = np.clip(rgb, 0, 255).astype(np.uint8)
    alpha = np.clip(alpha, 0, 1).astype(np.float32)
    
    eps = 1e-6
    alpha_safe = np.maximum(alpha, eps)
    true_rgb = (rgb_uint8.astype(np.float32) / alpha_safe[:, :, np.newaxis]).clip(0, 255).astype(np.uint8)
    
    mask = (alpha < source_threshold).astype(np.uint8)
    inpainted = cv2.inpaint(true_rgb, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
    
    alpha_3d = alpha[:, :, np.newaxis]
    result = true_rgb.astype(np.float32) * alpha_3d + inpainted.astype(np.float32) * (1 - alpha_3d)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def build_mask_from_alpha(
    alpha: np.ndarray,
    low_threshold: float = 0.55,
    high_threshold: float = 0.9,
    kernel_size: int = 3,
    min_region_size: int = 256,
    min_core_pixels: int = 64,
    min_core_ratio: float = 0.01,
    max_diffuse_area_ratio: float = 0.35,
) -> np.ndarray:
    """
    Build a conservative keep-mask from 3DGS alpha only.

    The mask uses hysteresis-style thresholding:
    - `high_threshold` defines trusted alpha support.
    - `low_threshold` defines candidate support around trusted regions.
    - large diffuse components with little trusted support are rejected.
    """

    alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * kernel_size + 1, 2 * kernel_size + 1)
    )

    candidate_mask = (alpha >= low_threshold).astype(np.uint8)
    core_mask = (alpha >= high_threshold).astype(np.uint8)

    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel)
    core_mask = cv2.morphologyEx(core_mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask)
    refined_mask = np.zeros_like(candidate_mask, dtype=np.uint8)

    image_area = alpha.shape[0] * alpha.shape[1]
    max_diffuse_area = max(int(image_area * max_diffuse_area_ratio), min_region_size)

    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_region_size:
            continue

        component = labels == lbl
        core_pixels = int(core_mask[component].sum())
        peak_alpha = float(alpha[component].max())

        if core_pixels == 0 or peak_alpha < high_threshold:
            continue

        core_ratio = core_pixels / float(area)
        if core_pixels < min_core_pixels and core_ratio < min_core_ratio:
            continue

        if area > max_diffuse_area and core_ratio < (min_core_ratio * 2.0):
            continue

        refined_mask[component] = 1

    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
    return refined_mask.astype(bool)


def fill_holes_with_alpha_blending(
    rgb: np.ndarray,
    alpha: np.ndarray,
    target_mask: np.ndarray,
    alpha_threshold: float = 0.99,
) -> np.ndarray:
    """
    Fill holes using alpha-aware blending.
    
    Args:
        rgb: [H, W, 3] RGB image from 3DGS
        alpha: [H, W] alpha values from 3DGS (0-1 floats)
        target_mask: [H, W] boolean target region to fill
        alpha_threshold: Alpha above this is considered solid for interpolation source
        
    Returns:
        filled_rgb: [H, W, 3] RGB with alpha-blended filling
    """
    solid_mask = alpha > alpha_threshold
    
    interpolated_rgb = fill_holes_with_interpolation(
        combined_rgb=rgb,
        rendered_mask=solid_mask,
        refined_mask=target_mask,
    )
    
    filled_rgb = rgb.copy()
    alpha_3d = alpha[:, :, np.newaxis]
    
    filled_rgb[target_mask] = (
        rgb[target_mask] * alpha_3d[target_mask] + 
        interpolated_rgb[target_mask] * (1 - alpha_3d[target_mask])
    )
    
    return filled_rgb
