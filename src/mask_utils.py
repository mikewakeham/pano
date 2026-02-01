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

