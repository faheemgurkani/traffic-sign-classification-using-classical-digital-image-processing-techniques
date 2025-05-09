import numpy as np



# def classify_sign(features):
#     """
#     Rule-based classification for traffic signs according to observed feature standards.
#     Features dict keys:
#       - avg_hue: average hue of segmented region (0–180 scale)
#       - circ: circularity (4π·area/perimeter²)
#       - ar: aspect ratio (width/height)
#       - extent: area/(bounding-box area)
#       - corners: number of vertices from contour approximation
#       - mask_color: 'red' or 'blue'
#     """
#     # hue = features['avg_hue']
#     circ = features['circ']
#     ar = features['ar']
#     # extent = features['extent']
#     corners = features['corners']
#     color = features['mask_color']
#     holes  = features.get('holes', 0)

#     # Speed Limit
#     if color == 'red' and (0.9 < circ <= 1.0) and abs(ar - 1.0) < 0.2:
#         if holes == 1:
#             return 0  # Speed Limit 20
#         elif holes == 2:
#             return 1  # Speed Limit 30
#         elif holes == 0:
#             return 17 # No Entry

#     # Yield Sign (triangle)
#     if color == 'red' and corners == 3 and (0.6 < circ < 0.75):
#         return 13

#     # Stop Sign (octagon)
#     if color == 'red' and corners == 8 and (0.85 < circ <= 1.0):
#         return 14

#     # No Entry (round with white minus)
#     if color == 'red' and (0.5 < circ < 0.7) and (12 <= corners <= 25):
#         return 17

#     # Keep Right (blue arrow, 4 vertices, elongated)
#     if color == 'blue' and (corners == 4 and ar > 1.2):
#         return 38

#     # fallback
#     return -1

def classify_sign(features):
    """
    Redesigned rules to match the observed feature patterns.
    Features:
      - avg_hue: average hue
      - circ: circularity
      - ar: aspect ratio (not used below)
      - extent: area / bbox_area
      - holes: count of dark connected blobs in ROI
    """

    hue   = features['avg_hue']
    circ  = features['circ']
    ext   = features['extent']
    holes = features['holes']

    # 1) KEEP RIGHT (38): the only blue sign → very high hue
    if hue > 120:
        return 38

    # 2) NO ENTRY (17): big red disc with a white bar → extremely low circularity & very low fill
    if circ < 0.3 and ext < 0.5:
        return 17

    # 3) SPEED LIMIT 20 (0) vs 30 (1): both are red circles, distinguish by holes in the “20” vs “30”
    if 0.4 < circ < 0.9 and ext > 0.6 and holes in (1, 2):
        return 0 if holes == 1 else 1

    # 4) STOP (14): red octagon → very high circularity on your mask (>0.85) and moderate fill
    if circ > 0.85 and ext > 0.6:
        return 14

    # 5) YIELD (13): red triangle → intermediate circularity and fairly high fill
    if 0.3 < circ < 0.85 and ext > 0.6:
        return 13

    # 6) anything else → unknown
    return -1
