import numpy as np
from scipy import ndimage



def rgb_to_hsv(img):
    # Input: img in RGB [0,255]
    rgb = img.astype('float32') / 255.0
    r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
    c_max = rgb.max(axis=2)
    c_min = rgb.min(axis=2)
    delta = c_max - c_min
    h = np.zeros_like(c_max)
    mask = delta != 0

    # Hue calculation
    idx = (c_max == r) & mask
    h[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6
    idx = (c_max == g) & mask
    h[idx] = (b[idx] - r[idx]) / delta[idx] + 2
    idx = (c_max == b) & mask
    h[idx] = (r[idx] - g[idx]) / delta[idx] + 4
    # h = h * 60
    # h[h < 0] += 360
    
    # convert to degrees
    h = h * 60
    h[h < 0] += 360
    # scaling to OpenCV's 0–180 hue range
    h = h / 2.0

    # Saturation
    s = np.where(c_max == 0, 0, delta / c_max)

    # Value
    v = c_max

    return np.stack([h, s*255, v*255], axis=2).astype(np.uint8)

def threshold_mask(hsv, color):
    h, s, v = hsv[...,0], hsv[...,1], hsv[...,2]
    
    if color == 'red':
        mask1 = (h <= 15) | (h >= 165)
        mask2 = (s >= 100) & (v >= 80)
    
        return (mask1 & mask2).astype(np.uint8)

    elif color == 'blue':
        mask = (h >= 100) & (h <= 130) & (s >= 100) & (v >= 80)
    
        return mask.astype(np.uint8)

    else:

        return np.zeros(h.shape, dtype=np.uint8)

def erode(mask, se=np.ones((3,3), dtype=np.uint8)):
    return ndimage.binary_erosion(mask, structure=se).astype(np.uint8)

def dilate(mask, se=np.ones((3,3), dtype=np.uint8)):
    return ndimage.binary_dilation(mask, structure=se).astype(np.uint8)

def opening(mask, se=np.ones((3,3), dtype=np.uint8)):
    return ndimage.binary_opening(mask, structure=se).astype(np.uint8)

def remove_small_components(mask, min_area=50):
    labeled, num = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled, range(1, num+1))
    mask_size = sizes >= min_area
    cleaned = mask.copy()

    for i, keep in enumerate(mask_size, start=1):

        if not keep:
            cleaned[labeled == i] = 0

    return cleaned

def fill_holes(mask):
    inv = 1 - mask
    filled = ndimage.binary_fill_holes(mask).astype(np.uint8)

    return filled