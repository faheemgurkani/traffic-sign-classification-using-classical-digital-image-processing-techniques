import numpy as np



# def classify_sign(features):
#     # features: dict with keys: 'avg_hue','circ','ar','extent','corners','mask_color'
#     hue = features['avg_hue']
#     circ = features['circ']
#     ar, extent = features['ar'], features['extent']
#     corners = features['corners']
#     color = features['mask_color']
    
#     # Example rules:
#     # allow “circular” signs with very few corners detected
#     if color == 'red' and circ > 0.6 and corners <= 2:
#         return 0   # Speed Limit 20
#     if color == 'red' and corners==8:
#         return 14  # Stop sign
#     if color == 'red' and ar>0.8 and ar<1.2 and corners==3:
#         return 13  # Yield sign
#     if color == 'blue' and circ<0.5 and ar>1.5:
#         return 38  # Keep right
    
#     # fallback
#     return -1

# def classify_sign(features):
#     hue = features['avg_hue']
#     circ = features['circ']
#     corners = features['corners']
#     ar = features['ar']
#     color = features['mask_color']

#     # Speed Limit: almost perfect circle, hardly any vertices
#     if color=='red' and circ>0.6 and corners<=1:
#         return 0
#     # Stop Sign: exactly 8 polygon corners
#     if color=='red' and corners==8:
#         return 14
#     # Yield Sign: triangle (3 vertices)
#     if color=='red' and corners==3:
#         return 13
#     # Keep Right: L-shaped arrow (4 vertices + aspect ratio)
#     if color=='blue' and corners==4 and ar>1.2:
#         return 38

#     return -1

def classify_sign(features):
    """
    Rule-based classification for traffic signs according to observed feature standards.
    Features dict keys:
      - avg_hue: average hue of segmented region (0–180 scale)
      - circ: circularity (4π·area/perimeter²)
      - ar: aspect ratio (width/height)
      - extent: area/(bounding-box area)
      - corners: number of vertices from contour approximation
      - mask_color: 'red' or 'blue'
    """
    hue = features['avg_hue']
    circ = features['circ']
    ar = features['ar']
    extent = features['extent']
    corners = features['corners']
    color = features['mask_color']

    # 0 & 1: Speed Limit signs (circular, ~17 vertices, ar≈1, circ≈0.75)
    if color == 'red' and (0.7 < circ < 0.85) and (15 <= corners <= 20) and (abs(ar - 1.0) < 0.2):
        return 0

    # 13: Yield Sign (triangle)
    if color == 'red' and corners == 3 and (0.6 < circ < 0.75):
        return 13

    # 14: Stop Sign (octagon)
    if color == 'red' and corners == 8 and (0.9 < circ <= 1.0):
        return 14

    # 17: Other red sign (e.g., No Entry: lower circularity, ~16–18 vertices)
    if color == 'red' and (0.5 < circ < 0.7) and (12 <= corners <= 25):
        return 17

    # 38: Keep Right (blue arrow, 4 vertices, elongated)
    if color == 'blue' and (corners == 4 and ar > 1.2):
        return 38

    # fallback
    return -1