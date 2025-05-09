import numpy as np



def affine_transform(img, matrix, output_shape):
    h_out, w_out = output_shape
    inv = np.linalg.inv(np.vstack([matrix, [0,0,1]]))
    out = np.zeros((h_out, w_out, img.shape[2]), dtype=img.dtype)

    for y in range(h_out):

        for x in range(w_out):
            src = inv.dot([x, y, 1])
            sx, sy = src[0], src[1]

            if 0 <= sx < img.shape[1]-1 and 0 <= sy < img.shape[0]-1:
                x0, y0 = int(sx), int(sy)
                dx, dy = sx-x0, sy-y0

                # bilinear interp
                for c in range(img.shape[2]):
                    v = (img[y0, x0, c] * (1-dx)*(1-dy) +
                         img[y0, x0+1, c] * dx*(1-dy) +
                         img[y0+1, x0, c] * (1-dx)*dy +
                         img[y0+1, x0+1, c] * dx*dy)

                    out[y, x, c] = int(v)

    return out

def resize(img, output_shape):
    """
    img: H×W×C or H×W mask; output_shape = (H_out, W_out)
    """
    H, W = img.shape[:2]
    H_out, W_out = output_shape
    
    # scale factors
    sx, sy = W_out / W, H_out / H
    
    # affine matrix: [ [sx, 0, 0], [0, sy, 0] ]
    M = np.array([[sx, 0, 0],
                  [0, sy, 0]])
    
    # reuse existing affine_transform
    from normalization import affine_transform
    
    if img.ndim == 2:
        # treat mask as single channel
        img3 = img[:,:,None]
        resized = affine_transform(img3, M, output_shape)
    
        return resized[:,:,0]
    else:
        return affine_transform(img, M, output_shape)