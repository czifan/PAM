from imports import *
import SimpleITK as sitk 
import albumentations as A
import cv2
from sklearn.metrics import jaccard_score, confusion_matrix
# from medpy.metric.binary import hd95, asd
import time 
from scipy.ndimage import label as scipy_label
from scipy.ndimage import sum as scipy_sum
from copy import deepcopy
import torch.nn.functional as F
from skimage import measure, data
from glob import glob 
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
from infer import *
import matplotlib.pyplot as plt

printer = print
args = build_args()
ds = build_dataset(args, printer=printer)
pred_norm_low_ps, pred_norm_high_ps = [], []
for i in range(len(ds)):
    if i == 5: break
    _, _, _, img_array, _, mask_array, _ = ds._read_image_mask(i)
    for k, mask_id in enumerate(ds.labels_dict):
        k_mask_array = (mask_array == mask_id).astype(np.uint8)
        k_mask_ind = np.where(k_mask_array)
        for z in list(set(k_mask_ind[0])):
            if np.sum(k_mask_array[z]) < 100:
                continue
            slice_img = img_array[z]
            slice_mask = k_mask_array[z]

            cropped_img, cropped_mask, _, _, _, _ = center_crop_only_support(slice_img, slice_mask)

            hu_values = cropped_img.flatten()
            foreground_hu_values = hu_values[np.where(cropped_mask.flatten() == 1)]
            norm_low, norm_high = np.percentile(foreground_hu_values, 0.5), np.percentile(foreground_hu_values, 99.5)
            gap_norm_low, gap_norm_high = 11111111, 11111111
            pred_norm_low, pred_norm_high = None, None 
            pred_norm_low_p, pred_norm_high_p = None, None

            for p in np.arange(0.1, 100, 0.1):
                norm_tmp = np.percentile(hu_values, p)
                if np.abs(norm_tmp - norm_low) < gap_norm_low:
                    gap_norm_low = np.abs(norm_tmp - norm_low)
                    pred_norm_low = norm_tmp
                    pred_norm_low_p = p
                if np.abs(norm_tmp - norm_high) < gap_norm_high:
                    gap_norm_high = np.abs(norm_tmp - norm_high)
                    pred_norm_high = norm_tmp
                    pred_norm_high_p = p
            pred_norm_low_ps.append(pred_norm_low_p)
            pred_norm_high_ps.append(pred_norm_high_p)
            print(f"norm_low={norm_low:.2f}, norm_high={norm_high:.2f}, pred_norm_low={pred_norm_low:.2f}({pred_norm_low_p:.1f}%), pred_norm_high={pred_norm_high:.2f}({pred_norm_high_p:.1f}%)")

print(np.argmax(pred_norm_low_ps), np.argmax(pred_norm_high_ps))
_, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].hist(pred_norm_low_ps, bins=100)
ax[1].hist(pred_norm_high_ps, bins=100)
plt.savefig("1111.jpg")
plt.close()
