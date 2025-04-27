import numpy as np
import cv2



def mean_filter(img, k=3):
    pad = k // 2
    h, w = img.shape[:2]
    out = np.zeros_like(img, dtype=np.float32)
    padded = np.pad(img, ((pad, pad), (pad, pad), (0,0)), mode='reflect')
    kernel = np.ones((k, k)) / (k*k)
    for y in range(h):

        for x in range(w):
            region = padded[y:y+k, x:x+k]
            out[y, x] = (region * kernel[...,None]).sum(axis=(0,1))

    return out.astype(img.dtype)

def gaussian_kernel1d(sigma, radius=None):
    
    if radius is None:
        radius = int(3 * sigma)
    
    x = np.arange(-radius, radius+1)
    g = np.exp(-(x**2)/(2*sigma**2))
    g /= g.sum()
    
    return g

def gaussian_filter(img, sigma=1.0):
    radius = int(3 * sigma)
    kernel = gaussian_kernel1d(sigma, radius)

    if img.ndim == 2:  # Grayscale
        padded = np.pad(img, ((radius, radius), (radius, radius)), mode='reflect')
        temp = np.zeros_like(padded, dtype=np.float32)

        # Convolving along x-axis
        for i in range(padded.shape[0]):
            temp[i, :] = np.convolve(padded[i, :], kernel, mode='same')

        temp = temp[radius:-radius, :]  # Remove vertical padding
        out = np.zeros_like(img, dtype=np.float32)

        # Convolving along y-axis
        for j in range(out.shape[1]):
            out[:, j] = np.convolve(temp[:, j], kernel, mode='same')

    elif img.ndim == 3:  # Color
        padded = np.pad(img, ((radius, radius), (radius, radius), (0, 0)), mode='reflect')
        temp = np.zeros_like(padded, dtype=np.float32)

        # Convolving along x-axis for each channel
        for c in range(3):
            
            for i in range(padded.shape[0]):
                temp[i, :, c] = np.convolve(padded[i, :, c], kernel, mode='same')

        temp = temp[radius:-radius, :, :]  # Remove vertical padding
        out = np.zeros_like(img, dtype=np.float32)

        # Convolving along y-axis for each channel
        for c in range(3):
            
            for j in range(out.shape[1]):
                out[:, j, c] = np.convolve(temp[:, j, c], kernel, mode='same')

    else:
        raise ValueError(f"Unsupported image dimensions: {img.ndim}")

    out = np.clip(out, 0, 255)

    return out.astype(img.dtype)

def median_filter(img, k=3):
    pad = k // 2
    h, w = img.shape[:2]
    out = np.zeros_like(img)
    padded = np.pad(img, ((pad,pad),(pad,pad),(0,0)), mode='reflect')

    for y in range(h):

        for x in range(w):
            region = padded[y:y+k, x:x+k]
            out[y, x] = np.median(region.reshape(-1, 3), axis=0)

    return out

def adaptive_median_filter(img, max_window_size=7):
    """
    Apply adaptive median filtering to an input image.

    Args:
        img (np.ndarray): Input RGB image.
        max_window_size (int): Maximum window size for adaptive behavior.

    Returns:
        np.ndarray: Filtered RGB image.
    """

    # Ensuring the input is copied safely
    img = img.copy()
    out = np.zeros_like(img)

    # For color images, apply filter channel-wise
    for ch in range(img.shape[2]):
        padded = np.pad(img[:, :, ch], pad_width=max_window_size//2, mode='reflect')
        filtered = np.zeros_like(img[:, :, ch])

        for i in range(filtered.shape[0]):
           
            for j in range(filtered.shape[1]):
                window_size = 3
                done = False

                while not done:
                    half = window_size // 2
                    window = padded[i:i + window_size, j:j + window_size]

                    z_min = int(np.min(window))
                    z_max = int(np.max(window))
                    z_med = int(np.median(window))
                    z_xy = int(padded[i + half, j + half])

                    A1 = z_med - z_min
                    A2 = z_med - z_max

                    if A1 > 0 and A2 < 0:
                        B1 = z_xy - z_min
                        B2 = z_xy - z_max
                  
                        if B1 > 0 and B2 < 0:
                            filtered[i, j] = z_xy
                        else:
                            filtered[i, j] = z_med
                  
                        done = True
                  
                    else:
                        window_size += 2
                  
                        if window_size > max_window_size:
                            filtered[i, j] = z_med
                            done = True

        out[:, :, ch] = filtered

    return out

def unsharp_mask(img, sigma=1.0, strength=1.5):
    blurred = gaussian_filter(img, sigma)
    detail = img.astype(np.float32) - blurred.astype(np.float32)
    out = img.astype(np.float32) + strength * detail
    out = np.clip(out, 0, 255)
    
    return out.astype(img.dtype)