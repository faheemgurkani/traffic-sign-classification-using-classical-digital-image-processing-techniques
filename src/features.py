import numpy as np
from scipy import ndimage
import cv2



def harris_corners(gray, window=3, k=0.04, thresh=1e-5):
    # gradients
    gx = ndimage.sobel(gray, axis=1)
    gy = ndimage.sobel(gray, axis=0)
    Ixx = gx*gx
    Iyy = gy*gy
    Ixy = gx*gy

    # response
    offset = window//2
    R = np.zeros_like(gray, dtype=np.float32)
    
    for y in range(offset, gray.shape[0]-offset):
    
        for x in range(offset, gray.shape[1]-offset):
            Sxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1].sum()
            Syy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1].sum()
            Sxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1].sum()
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            R[y, x] = det - k * (trace**2)

    # non-max suppression
    corners = (R > thresh)

    return np.count_nonzero(corners)

def count_vertices(mask):
    """Count polygon vertices from the largest contour in a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0
    
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    epsilon = 0.02 * peri
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    return len(approx)

def compute_circularity(mask):
    # mask: binary
    area = np.count_nonzero(mask)
    
    # perimeter via edge detection on mask
    eroded = ndimage.binary_erosion(mask)
    perimeter = area - np.count_nonzero(eroded)
    
    if perimeter == 0:
        return 0
    
    return (4 * np.pi * area) / (perimeter**2)

def aspect_ratio_and_extent(mask):
    y_idxs, x_idxs = np.nonzero(mask)

    if x_idxs.size == 0 or y_idxs.size == 0:
        return 0.0, 0.0  # Safe default values if mask is empty

    min_x, max_x = x_idxs.min(), x_idxs.max()
    min_y, max_y = y_idxs.min(), y_idxs.max()
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    area = mask.sum()
    bbox_area = width * height
    ar = width / float(height) if height > 0 else 0.0
    extent = area / float(bbox_area) if bbox_area > 0 else 0.0

    return ar, extent

def average_hue(hsv, mask):
    h = hsv[...,0]
    vals = h[mask.astype(bool)]
    
    return float(vals.mean()) if vals.size>0 else 0.0

def extract_roi_props(mask):
    """
    Compute ROI bounding box and image dims from binary mask.
    Returns: (width, height, x1, y1, x2, y2)
    where (x1,y1) is top-left and (x2,y2) bottom-right of ROI.
    """
    y_idxs, x_idxs = np.nonzero(mask)

    if x_idxs.size == 0 or y_idxs.size == 0:
        return 0, 0, 0, 0, 0, 0
    
    x1, x2 = x_idxs.min(), x_idxs.max()
    y1, y2 = y_idxs.min(), y_idxs.max()
    width = x2 - x1 + 1
    height = y2 - y1 + 1
    
    return width, height, x1, y1, x2, y2

def count_text_holes(gray, mask, thresh=80):
    """
    Count dark blobs (holes) inside the sign area.
    gray: 2D gray image, mask: 2D binary mask
    """
    # isolate potential text: dark pixels within mask
    # text = ((gray < thresh) & (mask.astype(bool))).astype(np.uint8)
    
    # try a more forgiving threshold for small images
    t = thresh or 100
    text = ((gray < t) & (mask.astype(bool))).astype(np.uint8)
    
    # remove tiny specks
    text = ndimage.binary_opening(text, structure=np.ones((3,3))).astype(np.uint8)
    
    # count connected components
    _, num = ndimage.label(text)
    
    return num