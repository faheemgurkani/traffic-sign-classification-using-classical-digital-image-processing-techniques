# import numpy as np
# from scipy import ndimage
# import cv2



# def convolve2d(image, kernel):
#     h, w = kernel.shape
#     pad_h, pad_w = h // 2, w // 2
#     padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
#     out = np.zeros_like(image, dtype=np.float32)

#     for i in range(image.shape[0]):

#         for j in range(image.shape[1]):
#             region = padded[i:i+h, j:j+w]
#             out[i, j] = np.sum(region * kernel)

#     return out

# def harris_corners(gray, window=3, k=0.04, thresh=1e-5):
#     # gradients
#     # gx = ndimage.sobel(gray, axis=1)
#     # gy = ndimage.sobel(gray, axis=0)

#     sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
#     sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
#     gx = convolve2d(gray, sobel_x)
#     gy = convolve2d(gray, sobel_y)

#     Ixx = gx*gx
#     Iyy = gy*gy
#     Ixy = gx*gy

#     # response
#     offset = window//2
#     R = np.zeros_like(gray, dtype=np.float32)
    
#     for y in range(offset, gray.shape[0]-offset):
    
#         for x in range(offset, gray.shape[1]-offset):
#             Sxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1].sum()
#             Syy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1].sum()
#             Sxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1].sum()
#             det = (Sxx * Syy) - (Sxy**2)
#             trace = Sxx + Syy
#             R[y, x] = det - k * (trace**2)

#     # non-max suppression
#     corners = (R > thresh)

#     return np.count_nonzero(corners)

# def count_vertices(mask):
#     """Count polygon vertices from the largest contour in a binary mask."""
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     if not contours:
#         return 0
    
#     cnt = max(contours, key=cv2.contourArea)
#     peri = cv2.arcLength(cnt, True)
#     epsilon = 0.02 * peri
#     approx = cv2.approxPolyDP(cnt, epsilon, True)
    
#     return len(approx)

# def compute_circularity(mask):
#     # mask: binary
#     area = np.count_nonzero(mask)
    
#     # perimeter via edge detection on mask
#     eroded = ndimage.binary_erosion(mask)
#     perimeter = area - np.count_nonzero(eroded)
    
#     if perimeter == 0:
#         return 0
    
#     return (4 * np.pi * area) / (perimeter**2)

# def aspect_ratio_and_extent(mask):
#     y_idxs, x_idxs = np.nonzero(mask)

#     if x_idxs.size == 0 or y_idxs.size == 0:
#         return 0.0, 0.0  # Safe default values if mask is empty

#     min_x, max_x = x_idxs.min(), x_idxs.max()
#     min_y, max_y = y_idxs.min(), y_idxs.max()
#     width = max_x - min_x + 1
#     height = max_y - min_y + 1
#     area = mask.sum()
#     bbox_area = width * height
#     ar = width / float(height) if height > 0 else 0.0
#     extent = area / float(bbox_area) if bbox_area > 0 else 0.0

#     return ar, extent

# def average_hue(hsv, mask):
#     h = hsv[...,0]
#     vals = h[mask.astype(bool)]
    
#     return float(vals.mean()) if vals.size>0 else 0.0

# def extract_roi_props(mask):
#     """
#     Compute ROI bounding box and image dims from binary mask.
#     Returns: (width, height, x1, y1, x2, y2)
#     where (x1,y1) is top-left and (x2,y2) bottom-right of ROI.
#     """
#     y_idxs, x_idxs = np.nonzero(mask)

#     if x_idxs.size == 0 or y_idxs.size == 0:
#         return 0, 0, 0, 0, 0, 0
    
#     x1, x2 = x_idxs.min(), x_idxs.max()
#     y1, y2 = y_idxs.min(), y_idxs.max()
#     width = x2 - x1 + 1
#     height = y2 - y1 + 1
    
#     return width, height, x1, y1, x2, y2

# def count_text_holes(gray, mask, thresh=80):
#     """
#     Count dark blobs (holes) inside the sign area.
#     gray: 2D gray image, mask: 2D binary mask
#     """
#     # isolate potential text: dark pixels within mask
#     # text = ((gray < thresh) & (mask.astype(bool))).astype(np.uint8)
    
#     # try a more forgiving threshold for small images
#     t = thresh or 100
#     text = ((gray < t) & (mask.astype(bool))).astype(np.uint8)
    
#     # remove tiny specks
#     text = ndimage.binary_opening(text, structure=np.ones((3,3))).astype(np.uint8)
    
#     # count connected components
#     _, num = ndimage.label(text)
    
#     return num

import numpy as np



def sobel_filters(img):
    Kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    gx = convolve(img, Kx)
    gy = convolve(img, Ky)

    return gx, gy

def convolve(img, kernel):
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    output = np.zeros_like(img, dtype=np.float32)

    for i in range(img.shape[0]):
        
        for j in range(img.shape[1]):
            region = padded[i:i + k_h, j:j + k_w]
            output[i, j] = np.sum(region * kernel)

    return output

def harris_corners(gray, window=3, k=0.04, thresh=1e-5):
    gx, gy = sobel_filters(gray)
    Ixx = gx * gx
    Iyy = gy * gy
    Ixy = gx * gy

    offset = window // 2
    R = np.zeros_like(gray, dtype=np.float32)

    for y in range(offset, gray.shape[0] - offset):
    
        for x in range(offset, gray.shape[1] - offset):
            Sxx = Ixx[y - offset:y + offset + 1, x - offset:x + offset + 1].sum()
            Syy = Iyy[y - offset:y + offset + 1, x - offset:x + offset + 1].sum()
            Sxy = Ixy[y - offset:y + offset + 1, x - offset:x + offset + 1].sum()
            det = Sxx * Syy - Sxy**2
            trace = Sxx + Syy
            R[y, x] = det - k * trace**2

    return np.count_nonzero(R > thresh)

def count_vertices(mask):
    from .segmentation import find_contours, arc_length, approx_polygon

    contours = find_contours(mask)
    
    if not contours:
        return 0

    cnt = max(contours, key=lambda c: np.abs(c[:, 0] * np.roll(c[:, 1], -1) - c[:, 1] * np.roll(c[:, 0], -1)).sum())
    peri = arc_length(cnt)
    epsilon = 0.02 * peri
    approx = approx_polygon(cnt, epsilon)

    return len(approx)

def compute_circularity(mask):
    area = np.count_nonzero(mask)
    edge = np.logical_xor(mask, binary_erosion(mask))
    perimeter = np.count_nonzero(edge)

    if perimeter == 0:
        return 0

    return (4 * np.pi * area) / (perimeter**2)

# def binary_erosion(mask, structure=np.ones((3, 3))):
#     pad_h, pad_w = structure.shape[0] // 2, structure.shape[1] // 2
#     padded = np.pad(mask, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
#     result = np.zeros_like(mask)

#     for i in range(mask.shape[0]):
    
#         for j in range(mask.shape[1]):
#             region = padded[i:i + 3, j:j + 3]
#             result[i, j] = np.all(region[structure == 1])

#     return result.astype(np.uint8)

def aspect_ratio_and_extent(mask):
    y_idxs, x_idxs = np.nonzero(mask)
   
    if x_idxs.size == 0 or y_idxs.size == 0:
        return 0.0, 0.0

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
    h = hsv[..., 0]
    vals = h[mask.astype(bool)]
   
    return float(vals.mean()) if vals.size > 0 else 0.0

def extract_roi_props(mask):
    y_idxs, x_idxs = np.nonzero(mask)
   
    if x_idxs.size == 0 or y_idxs.size == 0:
        return 0, 0, 0, 0, 0, 0

    x1, x2 = x_idxs.min(), x_idxs.max()
    y1, y2 = y_idxs.min(), y_idxs.max()
    width = x2 - x1 + 1
    height = y2 - y1 + 1

    return width, height, x1, y1, x2, y2

def count_text_holes(gray, mask, thresh=80):
    text = ((gray < thresh) & mask).astype(np.uint8)
    text = binary_opening(text)
  
    return count_connected_components(text)

def binary_opening(mask):
    return binary_erosion(binary_dilation(mask))

# def binary_dilation(mask, structure=np.ones((3, 3))):
#     pad_h, pad_w = structure.shape[0] // 2, structure.shape[1] // 2
#     padded = np.pad(mask, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
#     result = np.zeros_like(mask)

#     for i in range(mask.shape[0]):
  
#         for j in range(mask.shape[1]):
#             region = padded[i:i + 3, j:j + 3]
#             result[i, j] = np.any(region[structure == 1])

#     return result.astype(np.uint8)

def binary_dilation(mask, structure=np.ones((3, 3), dtype=np.uint8)):
    h, w = mask.shape
    sh, sw = structure.shape
    pad_y, pad_x = sh // 2, sw // 2
    padded = np.pad(mask, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=0)
    result = np.zeros_like(mask, dtype=np.uint8)

    for i in range(h):
        
        for j in range(w):
            region = padded[i:i+sh, j:j+sw]
            result[i, j] = np.any(region[structure == 1])

    return result

def binary_erosion(mask, structure=np.ones((3, 3), dtype=np.uint8)):
    h, w = mask.shape
    sh, sw = structure.shape
    pad_y, pad_x = sh // 2, sw // 2
    padded = np.pad(mask, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=0)
    result = np.zeros_like(mask, dtype=np.uint8)

    for i in range(h):
        
        for j in range(w):
            region = padded[i:i+sh, j:j+sw]
            result[i, j] = np.all(region[structure == 1])

    return result

# def binary_closing(mask, structure=np.ones((3, 3))):
#     return binary_erosion(binary_dilation(mask, structure), structure)

def binary_closing(mask, structure=np.ones((3, 3), dtype=np.uint8)):
    return binary_erosion(binary_dilation(mask, structure), structure)

def count_connected_components(mask):
    visited = np.zeros_like(mask, dtype=bool)
    count = 0
    h, w = mask.shape

    def dfs(x, y):
        stack = [(x, y)]
        
        while stack:
            i, j = stack.pop()
    
            if 0 <= i < h and 0 <= j < w and mask[i, j] and not visited[i, j]:
                visited[i, j] = True
    
                for dx in [-1, 0, 1]:
    
                    for dy in [-1, 0, 1]:
    
                        if dx != 0 or dy != 0:
                            stack.append((i + dx, j + dy))

    for i in range(h):
    
        for j in range(w):
    
            if mask[i, j] and not visited[i, j]:
                dfs(i, j)
                count += 1

    return count

def label_connected_components(mask):
    """
    Labels connected components in a binary mask using 4-connectivity.
    Returns: (labeled_mask, num_labels)
    """
    from collections import defaultdict

    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    label = 1
    parent = dict()

    def find(x):
        root = x
        
        while parent[root] != root:
            root = parent[root]
        
        while parent[x] != x:
            x, parent[x] = parent[x], root
        
        return root

    def union(x, y):
        root_x, root_y = find(x), find(y)
        
        if root_x != root_y:
            parent[root_y] = root_x

    # First pass
    for y in range(h):
        
        for x in range(w):
        
            if mask[y, x] == 0:
                continue
        
            neighbors = []
        
            if x > 0 and labels[y, x-1] > 0:
                neighbors.append(labels[y, x-1])
        
            if y > 0 and labels[y-1, x] > 0:
                neighbors.append(labels[y-1, x])
        
            if neighbors:
                min_label = min(neighbors)
                labels[y, x] = min_label
        
                for n in neighbors:
        
                    if n != min_label:
                        union(min_label, n)
            else:
                labels[y, x] = label
                parent[label] = label
                label += 1

    # Second pass
    label_map = {}
    new_label = 1
    
    for y in range(h):
    
        for x in range(w):
    
            if labels[y, x] > 0:
                root = find(labels[y, x])
    
                if root not in label_map:
                    label_map[root] = new_label
                    new_label += 1
    
                labels[y, x] = label_map[root]

    return labels, new_label - 1
