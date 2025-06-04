import numpy as np
import cv2

import os
import csv
import tqdm

from image_io import read_image
from filters import mean_filter, gaussian_filter, median_filter, adaptive_median_filter, unsharp_mask
from segmentation import rgb_to_hsv, threshold_mask, erode, dilate, opening, remove_small_components, fill_holes
from edge_detection import canny_edge
from normalization import affine_transform
from features import harris_corners, compute_circularity, aspect_ratio_and_extent, average_hue, extract_roi_props, count_vertices
from classifier import classify_sign
from evaluation import compute_and_save_metrics



def process_image(path):
    img = read_image(path)

    # from normalization import resize 

    # img = resize(img, (200, 200))
    
    if img is None:
        print(f"\n[Warning] Could not read image: {path}")
    
        return None, None

    height, width = img.shape[:2]

    # Filtering
    f = gaussian_filter(img, sigma=1.0)
    # skip heavy median filters (they break thin borders/text)
    sharp = unsharp_mask(f, sigma=1.0, strength=1.5)
    
    # Segmentation
    hsv = rgb_to_hsv(sharp)
    mask_r = threshold_mask(hsv, 'red')
    mask_b = threshold_mask(hsv, 'blue')
    mask = mask_r if mask_r.sum() > mask_b.sum() else mask_b
    # m = opening(mask)
    # m = remove_small_components(m, min_area=100)
    # m = fill_holes(m)

    # reconnect then clean
    # Closing (dilate → erode)
    se_close = np.ones((5,5), dtype=np.uint8)
    m = dilate(mask, se_close)
    m = erode (m, se_close) 
    
    # Opening to kill tiny speckles
    m = opening(m, se=np.ones((3,3), dtype=np.uint8))
    m = remove_small_components(m, min_area=200)
    m = fill_holes(m)
    
    # Edge detection
    gray = cv2.cvtColor(sharp, cv2.COLOR_RGB2GRAY)
    edges = canny_edge(gray)

    # New: count text holes inside the segmented mask
    from features import count_text_holes
    
    hole_count = count_text_holes(gray, m)
    
    # Normalization (identity here; placeholder)
    norm = sharp

    features = {}

    w, hgt, x1, y1, x2, y2 = extract_roi_props(m)

    features['width'], features['height'] = w, hgt
    features['roi_x1'], features['roi_y1'] = x1, y1
    features['roi_x2'], features['roi_y2'] = x2, y2

    # only look at the ROI in the grayscale image, masked by your segmentation
    gray_full = cv2.cvtColor(norm, cv2.COLOR_RGB2GRAY)
    # crop to ROI, then mask out anything outside the sign
    roi_gray = gray_full[y1:y2+1, x1:x2+1]
    roi_mask = m[y1:y2+1, x1:x2+1].astype(bool)
    # zero out background
    roi_gray_masked = roi_gray * roi_mask
    features['corners'] = harris_corners(roi_gray_masked)

    # # Crop and resize ROI to 200×200   
    # roi_rgb  = norm[y1:y2+1, x1:x2+1]
    # roi_mask = m [y1:y2+1, x1:x2+1] 
    # roi_rgb200  = resize(roi_rgb,  (200,200))
    # roi_mask200 = resize(roi_mask, (200,200))   
    
    # # recompute gray+masked ROI for corners
    # roi_gray200 = cv2.cvtColor(roi_rgb200, cv2.COLOR_RGB2GRAY)
    # features['corners'] = harris_corners(roi_gray200 * (roi_mask200))
    
    features['circ'] = compute_circularity(m)
    features['ar'], features['extent'] = aspect_ratio_and_extent(m)
    features['avg_hue'] = average_hue(hsv, m)
    features['mask_color'] = 'red' if mask is mask_r else 'blue'
    features['holes'] = hole_count
    
    # Classification
    cid = classify_sign(features)

    # # For, testing
    # print(f"cid: {cid}, features: {features}")
    # print()
    
    return cid, features

def run_pipeline(selected_csv, data_root, output_results):
    results_dir = os.path.dirname(output_results)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

        print(f"[Info] Created directory: {results_dir}")

    total = sum(1 for _ in open(selected_csv)) - 1  # excluding the header

    processed = 0

    with open(selected_csv) as f, open(output_results, 'w', newline='') as out:
        reader = csv.DictReader(f)
        writer = csv.writer(out)

        writer.writerow(['filename', 'ground_truth', 'predicted', 'correct', 'width', 'height', 'roi_x1', 'roi_y1', 'roi_x2', 'roi_y2'])

        print("\n[Info] Starting pipeline...")
        print(f"\n[Info] Total images to process: {total}")
        print()
        for idx, row in (enumerate(reader)):
            
            if idx == 599:
                break

            rel = row['Path']
            gt = int(row['ClassId'])
            path = os.path.join(data_root, rel)
            
            pred, feats = process_image(path)

            if pred is None:
                pred = -1  # if image couldn't be processed, mark as unknown class

            correct = int(pred == gt)
            
            writer.writerow([
                rel, gt, pred, correct,
                feats.get('width'), feats.get('height'),
                feats.get('roi_x1'), feats.get('roi_y1'),
                feats.get('roi_x2'), feats.get('roi_y2')
            ])
            
            processed += 1
            
            print(f"[Progress] Processed {processed}/{total}: {rel} (GT: {gt}, Predicted: {pred})")
            # print()

        print("\n[Info] Pipeline completed.")
    print(f"\n[Done] Results written to {output_results}")

    compute_and_save_metrics(output_results, results_dir)



if __name__ == '__main__':
    selected_csv = '../data/selected_round_robin.csv'
    data_root = '../data/selected'
    output_results = '../results/results.csv'
    run_pipeline(selected_csv, data_root, output_results)