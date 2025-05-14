# import numpy as np
# from scipy import ndimage



# def rgb_to_hsv(img):
#     # Input: img in RGB [0,255]
#     rgb = img.astype('float32') / 255.0
#     r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
#     c_max = rgb.max(axis=2)
#     c_min = rgb.min(axis=2)
#     delta = c_max - c_min
#     h = np.zeros_like(c_max)
#     mask = delta != 0

#     # Hue calculation
#     idx = (c_max == r) & mask
#     h[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6
#     idx = (c_max == g) & mask
#     h[idx] = (b[idx] - r[idx]) / delta[idx] + 2
#     idx = (c_max == b) & mask
#     h[idx] = (r[idx] - g[idx]) / delta[idx] + 4
#     # h = h * 60
#     # h[h < 0] += 360
    
#     # convert to degrees
#     h = h * 60
#     h[h < 0] += 360
#     # scaling to OpenCV's 0–180 hue range
#     h = h / 2.0

#     # Saturation
#     s = np.where(c_max == 0, 0, delta / c_max)

#     # Value
#     v = c_max

#     return np.stack([h, s*255, v*255], axis=2).astype(np.uint8)

# def threshold_mask(hsv, color):
#     h, s, v = hsv[...,0], hsv[...,1], hsv[...,2]
    
#     if color == 'red':
#         # mask1 = (h <= 15) | (h >= 165)
#         # mask2 = (s >= 100) & (v >= 80)

#         # mask1 = (h <= 20) | (h >= 160)
#         # mask2 = (s >= 60) & (v >= 60)
        
#         mask1 = (h <= 25) | (h >= 155)
#         mask2 = (s >= 50) & (v >= 50)
    
#         mask = mask1 & mask2

#     elif color == 'blue':
#         # mask = (h >= 100) & (h <= 130) & (s >= 100) & (v >= 80)

#         # mask = (h >= 90) & (h <= 140) & (s >= 60) & (v >= 60)
        
#         mask = (h >= 85) & (h <= 145) & (s >= 50) & (v >= 50)

#     else:
#         mask = np.zeros_like(h, dtype=bool)

#         # return np.zeros(h.shape, dtype=np.uint8)

#     # return mask.astype(np.uint8)
    
#     # first close small gaps in the border, then return
#     from scipy import ndimage
    
#     mask = ndimage.binary_closing(mask, structure=np.ones((5,5)))
    
#     return mask.astype(np.uint8)

# def erode(mask, se=np.ones((3,3), dtype=np.uint8)):
#     return ndimage.binary_erosion(mask, structure=se).astype(np.uint8)

# def dilate(mask, se=np.ones((3,3), dtype=np.uint8)):
#     return ndimage.binary_dilation(mask, structure=se).astype(np.uint8)

# def opening(mask, se=np.ones((3,3), dtype=np.uint8)):
#     return ndimage.binary_opening(mask, structure=se).astype(np.uint8)

# def remove_small_components(mask, min_area=50):
#     labeled, num = ndimage.label(mask)
#     sizes = ndimage.sum(mask, labeled, range(1, num+1))
#     mask_size = sizes >= min_area
#     cleaned = mask.copy()

#     for i, keep in enumerate(mask_size, start=1):

#         if not keep:
#             cleaned[labeled == i] = 0

#     return cleaned

# def fill_holes(mask):
#     inv = 1 - mask
#     filled = ndimage.binary_fill_holes(mask).astype(np.uint8)

#     return filled

# import numpy as np
# import cv2



# def rgb_to_hsv(img):
#     rgb = img.astype('float32') / 255.0
#     r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
#     c_max = rgb.max(axis=2)
#     c_min = rgb.min(axis=2)
#     delta = c_max - c_min
#     h = np.zeros_like(c_max)
#     mask = delta != 0
#     idx = (c_max == r) & mask
#     h[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6
#     idx = (c_max == g) & mask
#     h[idx] = (b[idx] - r[idx]) / delta[idx] + 2
#     idx = (c_max == b) & mask
#     h[idx] = (r[idx] - g[idx]) / delta[idx] + 4
#     h = h * 60
#     h[h < 0] += 360
#     h = h / 2.0
#     s = np.where(c_max == 0, 0, delta / c_max)
#     v = c_max

#     return np.stack([h, s * 255, v * 255], axis=2).astype(np.uint8)

# def threshold_mask(hsv, color):
#     h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

#     if color == 'red':
#         mask1 = (h <= 25) | (h >= 155)
#         mask2 = (s >= 50) & (v >= 50)
#         mask = mask1 & mask2
#     elif color == 'blue':
#         mask = (h >= 85) & (h <= 145) & (s >= 50) & (v >= 50)
#     else:
#         mask = np.zeros_like(h, dtype=bool)

#     mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

#     return mask

# def erode(mask, se=np.ones((3, 3), dtype=np.uint8)):
#     return cv2.erode(mask, se)

# def dilate(mask, se=np.ones((3, 3), dtype=np.uint8)):
#     return cv2.dilate(mask, se)

# def opening(mask, se=np.ones((3, 3), dtype=np.uint8)):
#     return cv2.morphologyEx(mask, cv2.MORPH_OPEN, se)

# def remove_small_components(mask, min_area=50):
#     num, labeled = cv2.connectedComponents(mask.astype(np.uint8))
#     output = np.zeros_like(mask)

#     for i in range(1, num):
#         component = (labeled == i)

#         if np.sum(component) >= min_area:
#             output[component] = 1

#     return output

# def fill_holes(mask):
#     mask_inv = np.logical_not(mask).astype(np.uint8)
#     h, w = mask.shape
#     flood_fill = np.zeros((h+2, w+2), np.uint8)
#     mask_filled = mask.copy()
#     cv2.floodFill(mask_filled, flood_fill, (0, 0), 1)
#     mask_filled_inv = cv2.bitwise_not(mask_filled)

#     return mask | (mask_filled_inv & 1)

import numpy as np
from features import binary_dilation, binary_erosion, label_connected_components, binary_closing



# def rgb_to_hsv(img):
#     rgb = img.astype('float32') / 255.0
#     r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
#     c_max = rgb.max(axis=2)
#     c_min = rgb.min(axis=2)
#     delta = c_max - c_min
#     h = np.zeros_like(c_max)
#     mask = delta != 0
#     idx = (c_max == r) & mask
#     h[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6
#     idx = (c_max == g) & mask
#     h[idx] = (b[idx] - r[idx]) / delta[idx] + 2
#     idx = (c_max == b) & mask
#     h[idx] = (r[idx] - g[idx]) / delta[idx] + 4
#     h = h * 60
#     h[h < 0] += 360
#     h = h / 2.0
#     s = np.where(c_max == 0, 0, delta / c_max)
#     v = c_max

#     return np.stack([h, s * 255, v * 255], axis=2).astype(np.uint8)

def rgb_to_hsv(img):
    rgb = img.astype('float32') / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    c_max = rgb.max(axis=2)
    c_min = rgb.min(axis=2)
    delta = c_max - c_min
    h = np.zeros_like(c_max)
    mask = delta != 0
    idx = (c_max == r) & mask
    h[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6
    idx = (c_max == g) & mask
    h[idx] = (b[idx] - r[idx]) / delta[idx] + 2
    idx = (c_max == b) & mask
    h[idx] = (r[idx] - g[idx]) / delta[idx] + 4
    h = h * 60
    h[h < 0] += 360
    h = h / 2.0
    
    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-8
    s = np.where(c_max == 0, 0, delta / (c_max + epsilon))  # Add epsilon to c_max
    v = c_max

    return np.stack([h, s * 255, v * 255], axis=2).astype(np.uint8)

def threshold_mask(hsv, color):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    if color == 'red':
        mask1 = (h <= 25) | (h >= 155)
        mask2 = (s >= 50) & (v >= 50)
        mask = mask1 & mask2
    elif color == 'blue':
        mask = (h >= 85) & (h <= 145) & (s >= 50) & (v >= 50)
    else:
        mask = np.zeros_like(h, dtype=bool)

    return binary_closing(mask.astype(np.uint8), np.ones((5, 5), dtype=np.uint8))

def erode(mask, se=np.ones((3, 3), dtype=np.uint8)):
    return binary_erosion(mask, se)

def dilate(mask, se=np.ones((3, 3), dtype=np.uint8)):
    return binary_dilation(mask, se)

def opening(mask, se=np.ones((3, 3), dtype=np.uint8)):
    return binary_erosion(binary_dilation(mask, se), se)

def remove_small_components(mask, min_area=50):
    labeled, count = label_connected_components(mask)
    output = np.zeros_like(mask)

    for i in range(1, count + 1):
        component = labeled == i
        
        if np.sum(component) >= min_area:
            output[component] = 1

    return output

def fill_holes(mask):
    filled = mask.copy()
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)

    # def flood(x, y):
    #     if x < 0 or x >= h or y < 0 or y >= w or mask[x, y] or visited[x, y]:
    #         return
    #     visited[x, y] = True
    #     for dx in [-1, 0, 1]:
    #         for dy in [-1, 0, 1]:
    #             flood(x + dx, y + dy)

    def flood(x, y):
        if x < 0 or x >= h or y < 0 or y >= w or mask[x, y] or visited[x, y]:
            return

        stack = [(x, y)]

        while stack:
            cx, cy = stack.pop()

            if cx < 0 or cx >= h or cy < 0 or cy >= w or mask[cx, cy] or visited[cx, cy]:
                continue

            visited[cx, cy] = True

            for dx in [-1, 0, 1]:

                for dy in [-1, 0, 1]:
                    stack.append((cx + dx, cy + dy))

    flood(0, 0)

    return mask | (~visited).astype(np.uint8)
