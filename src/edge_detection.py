import numpy as np
from scipy import ndimage



def sobel_gradients(gray):
    Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
    Ky = Kx.T
    gx = ndimage.convolve(gray.astype(np.float32), Kx)
    gy = ndimage.convolve(gray.astype(np.float32), Ky)

    return gx, gy

def non_max_suppression(mag, angle):
    H, W = mag.shape
    out = np.zeros((H,W), dtype=np.float32)
    angle = angle * 180.0 / np.pi
    angle[angle < 0] += 180
    
    for i in range(1, H-1):
    
        for j in range(1, W-1):
            q = 255; r = 255
    
            # angle 0
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = mag[i, j+1]; r = mag[i, j-1]
    
            # angle 45
            elif (22.5 <= angle[i,j] < 67.5):
                q = mag[i-1, j+1]; r = mag[i+1, j-1]
    
            # 90
            elif (67.5 <= angle[i,j] < 112.5):
                q = mag[i-1, j]; r = mag[i+1, j]
    
            # 135
            elif (112.5 <= angle[i,j] < 157.5):
                q = mag[i-1, j-1]; r = mag[i+1, j+1]
    
            if (mag[i,j] >= q) and (mag[i,j] >= r):
                out[i,j] = mag[i,j]
    
    return out

def double_threshold_and_hysteresis(nms, low, high):
    strong = (nms >= high)
    weak = ((nms >= low) & (nms < high))
    result = np.zeros_like(nms, dtype=np.uint8)
    result[strong] = 255
    H, W = nms.shape
    
    # Hysteresis
    for i in range(1, H-1):
    
        for j in range(1, W-1):
    
            if weak[i,j] and ((result[i+1,j] == 255) or (result[i-1,j] == 255)
                              or (result[i,j+1] == 255) or (result[i,j-1] == 255)
                              or (result[i+1,j+1] == 255) or (result[i-1,j-1] == 255)
                              or (result[i+1,j-1] == 255) or (result[i-1,j+1] == 255)):
    
                result[i,j] = 255
    
    return result

def canny_edge(gray, low=50, high=150):
    gx, gy = sobel_gradients(gray)
    mag = np.hypot(gx, gy)
    angle = np.arctan2(gy, gx)
    nms = non_max_suppression(mag, angle)
    edges = double_threshold_and_hysteresis(nms, low, high)
    
    return edges