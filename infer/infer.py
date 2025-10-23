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
from multiprocessing import Pool
from math import ceil

transform = A.Compose([
    A.Resize(width=224, height=224, p=1.0),
    A.Normalize(mean=[0.5], std=(0.5)),
    ToTensorV2(),
])
transform_2down = A.Compose([
    A.Resize(width=112, height=112, p=1.0),
    A.Normalize(mean=[0.5], std=(0.5)),
    ToTensorV2(),
])


def compute_dynamic_window(support_img, support_mask):
    foreground_pixels = support_img[np.where(support_mask > 0)].reshape(-1)
    img_min, img_max = np.percentile(foreground_pixels, 0.5), np.percentile(foreground_pixels, 99.5)
    window_center = int((img_min + img_max) / 2.0)
    window_width = int(img_max - img_min)
    return window_center, window_width

def compute_dynamic_internal(support_img, support_mask):
    foreground_pixels = support_img[np.where(support_mask > 0)].reshape(-1)
    img_min, img_max = np.percentile(foreground_pixels, 0.5), np.percentile(foreground_pixels, 99.5)
    return img_min, img_max

# def window_transform(img, window_center=50, window_width=350):
#     img_min = window_center - window_width // 2
#     img_max = window_center + window_width // 2
#     windowed_img = np.clip(img, img_min, img_max)
#     return windowed_img

def window_transform(img, window_center=50, window_width=350):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    windowed_img = np.clip(img, img_min, img_max)
    return windowed_img

def internal_transform(img, minv, maxv):
    return np.clip(img, minv, maxv)

def normalize(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255.0
    image = image.astype(np.uint8)
    return image
    
def crop(image, x1, y1, x2, y2, crop_size):
    cropped_image = image[y1:y2, x1:x2]
    if cropped_image.shape[0] < crop_size[0] or cropped_image.shape[1] < crop_size[1]:
        cropped_image = cv2.copyMakeBorder(cropped_image, top=0, bottom=crop_size[0] - cropped_image.shape[0],
                                            left=0, right=crop_size[1] - cropped_image.shape[1],
                                            borderType=cv2.BORDER_CONSTANT, value=0)
    return cropped_image

# def center_crop(image, mask, query_images, crop_size=(224, 224)):
#     H, W = mask.shape 
#     center_y, center_x = np.argwhere(mask > 0).mean(axis=0).astype(np.int16)
#     x1 = max(center_x - crop_size[1] // 2, 0)
#     y1 = max(center_y - crop_size[0] // 2, 0)
#     x2 = min(x1 + crop_size[1], W)
#     y2 = min(y1 + crop_size[0], H)
#     cropped_image = crop(image, x1, y1, x2, y2, crop_size)
#     cropped_mask = crop(mask, x1, y1, x2, y2, crop_size)
#     cropped_query_images = [crop(query_image, x1, y1, x2, y2, crop_size) for query_image in query_images]
#     return cropped_image, cropped_mask, cropped_query_images, x1, y1, x2, y2

def crop_with_resize(image, x1, y1, x2, y2, crop_size, is_mask):
    cropped_image = image[y1:y2, x1:x2]
    #cropped_image = cv2.resize(cropped_image, (crop_size[1], crop_size[0]), interpolation=cv2.INTER_LINEAR)
    if cropped_image.shape[0] < crop_size[0] or cropped_image.shape[1] < crop_size[1]:
        cropped_image = cv2.copyMakeBorder(cropped_image, top=0, bottom=crop_size[0] - cropped_image.shape[0],
                                            left=0, right=crop_size[1] - cropped_image.shape[1],
                                            borderType=cv2.BORDER_CONSTANT, value=0)
    return cropped_image

def center_crop(image, mask, query_images, dynamic_crop_ratio=1.5, min_crop_size=16):
    H, W = mask.shape 
    inds = np.where(mask)
    y1, x1, y2, x2 = np.min(inds[0]), np.min(inds[1]), np.max(inds[0]), np.max(inds[1])
    center_y = int((y1 + y2) / 2.0)
    center_x = int((x1 + x2) / 2.0)
    h = y2 - y1
    w = x2 - x1
    crop_size = int(dynamic_crop_ratio * max(h, w))
    crop_size = max(crop_size, min_crop_size)
    crop_size = (crop_size, crop_size)
    x1 = max(center_x - crop_size[1] // 2, 0)
    y1 = max(center_y - crop_size[0] // 2, 0)
    x2 = min(x1 + crop_size[1], W)
    y2 = min(y1 + crop_size[0], H)
    cropped_image = crop_with_resize(image, x1, y1, x2, y2, crop_size, False)
    cropped_mask = crop_with_resize(mask, x1, y1, x2, y2, crop_size, True)
    cropped_query_images = [crop_with_resize(query_image, x1, y1, x2, y2, crop_size, False) for query_image in query_images]
    return cropped_image, cropped_mask, cropped_query_images, x1, y1, x2, y2

def center_crop_only_support(image, mask, sample_size=112, min_crop_size=16):
    H, W = mask.shape 
    inds = np.where(mask)
    y1, x1, y2, x2 = np.min(inds[0]), np.min(inds[1]), np.max(inds[0]), np.max(inds[1])
    center_y = int((y1 + y2) / 2.0)
    center_x = int((x1 + x2) / 2.0)
    h = y2 - y1
    w = x2 - x1
    # crop_size = int(1.0 * max(h, w))
    # crop_size = max(crop_size, min_crop_size)
    crop_size = (h, w)
    # crop_size = (h, w)
    x1 = max(center_x - crop_size[1] // 2, 0)
    y1 = max(center_y - crop_size[0] // 2, 0)
    x2 = min(x1 + crop_size[1], W)
    y2 = min(y1 + crop_size[0], H)
    # cropped_image = crop_with_resize(image, x1, y1, x2, y2, crop_size, False)
    # cropped_mask = crop_with_resize(mask, x1, y1, x2, y2, crop_size, True)
    cropped_image = image[y1:y2, x1:x2].astype(np.int16)
    if cropped_image.size == 0:
        return None, None, None, None, None, None
    cropped_image = cv2.resize(cropped_image, (sample_size, sample_size))
    cropped_mask = mask[y1:y2, x1:x2]
    cropped_mask = cv2.resize(cropped_mask, (sample_size, sample_size), interpolation=cv2.INTER_NEAREST)
    # _, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].imshow(cropped_image, cmap='gray')
    # ax[1].imshow(cropped_mask, cmap='gray')
    # plt.savefig(f"1111/{np.random.randint(1000)}.jpg")
    # plt.close()
    return cropped_image, cropped_mask, x1, y1, x2, y2

def pad_to_match_max_length(tensors_list):
    max_length = max(t.size(0) for t in tensors_list)
    padded_tensors = []
    for t in tensors_list:
        length = t.size(0)
        if length < max_length:
            repeat_times = max_length - length
            last_element_expanded = t[-1:].expand(repeat_times, *t.size()[1:])
            padded_tensor = torch.cat([t, last_element_expanded], dim=0)
        else:
            padded_tensor = t
        padded_tensors.append(padded_tensor)
    return padded_tensors

def remove_false_positive(mask, init_mask_array):
    labeled_mask, num_features = scipy_label(mask)
    if num_features == 0:
        return mask
    init_mask_nonzero = init_mask_array > 0
    for i in range(1, num_features + 1):
        region = labeled_mask == i
        if not np.any(region & init_mask_nonzero):
            mask[region] = 0
    return mask


def _process(percentile_low, percentile_high, foreground_pixels, support_x):
    img_min, img_max = np.percentile(foreground_pixels, percentile_low), np.percentile(foreground_pixels, percentile_high)
    window_center = int((img_min + img_max) / 2.0)
    window_width = int(img_max - img_min)
    query_x = normalize(window_transform(support_x, window_center, window_width))
    query_x = transform(image=query_x)["image"] # (1, 224, 224)
    return query_x


def infer_by_box2seg(args, support_categorie, box2seg, support_z, image, mask, dynamic_crop_ratio, device, printer):
    box2seg.eval()
    sample_size = 112
    support_x, support_y, x1, y1, x2, y2 = center_crop_only_support(image, mask, sample_size)
    if support_x is None:
        return None
    h, w = y2-y1, x2-x1
    # sample_size = 112
    pixels = support_x.reshape(-1)
    # pixels = cv2.resize(support_x, (sample_size, sample_size)).reshape(-1) # 降低采样，以减低下面取前景的复杂度
    query_xs = []
    tmps = []
    for percentile_low in np.arange(args.box2seg_percentile_low_start, args.box2seg_percentile_low_end, args.box2seg_percentile_low_interval):
        for percentile_high in np.arange(args.box2seg_percentile_high_start, args.box2seg_percentile_high_end, args.box2seg_percentile_high_interval):
            tmps.append(f"{percentile_low:.1f}_{percentile_high:.1f}")
            img_min, img_max = np.percentile(pixels, percentile_low), np.percentile(pixels, percentile_high)
            window_center = int((img_min + img_max) / 2.0)
            window_width = int(img_max - img_min)
            query_x = normalize(window_transform(support_x, window_center, window_width))
            query_x = transform(image=query_x)["image"] # (1, 224, 224)
            query_xs.append(query_x)
    query_xs = torch.stack(query_xs, dim=0).unsqueeze(dim=1) # (B, 1, 1, 224, 224)
    all_preds = []
    batch_size = 512
    foreground_pixels = []
    for i in range(0, query_xs.size(0), batch_size):
        batch = {
            "query_xs": query_xs[i:i+batch_size].float().to(device) # (B, 1, 1, 224, 224)
        }
        with torch.no_grad():
            pred_dict = box2seg(batch)
            preds = pred_dict["predictions"][1] # (BNq, 2, 112, 112)
            preds = torch.argmax(preds, dim=1) # (BNq, 112, 112)
            all_preds.append(preds)
    all_preds = torch.cat(all_preds, dim=0)
    preds = all_preds.detach().cpu().numpy()



    foreground_pixels = []
    for pred in preds:
        if pred.sum() < 10: continue
        foreground_pixels.extend(pixels[np.where(pred.reshape(-1) > 0)])
    if len(foreground_pixels) == 0:
        window_center = 50
        window_width = 350
    else:
        img_min, img_max = np.percentile(foreground_pixels, 0.5), np.percentile(foreground_pixels, 99.5)
        window_center = int((img_min + img_max) / 2.0)
        window_width = int(img_max - img_min)
    support_x = normalize(window_transform(support_x, window_center, window_width))
    support_x = transform(image=support_x)["image"] # (1, 224, 224)
    batch = {
        "query_xs": support_x.unsqueeze(dim=0).unsqueeze(dim=0).float().to(device) # (1, 1, 1, 224, 224)
    }
    with torch.no_grad():
        pred_dict = box2seg(batch)
        preds = [F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=True) for pred in pred_dict["predictions"]]
        preds = torch.mean(torch.stack(preds, dim=0), dim=0)
        preds = torch.argmax(preds, dim=1) # (BNq, H, W)
    #assert preds.shape[0] == 1, preds.shape 
    pred_support_y = np.zeros_like(mask).astype(mask.dtype)
    if preds.max() == 0:
        pred_support_y[int(y1):int(y2), int(x1):int(x2)] = 1
    else:
        pred_y = preds[0].detach().cpu().numpy().astype(mask.dtype)
        pred_support_y[int(y1):int(y2), int(x1):int(x2)] = pred_y
    printer(f"[Box2seg] {support_categorie} support_z={support_z}, dice={dice_coefficient(pred_support_y, mask):.4f}, mean={np.mean(pixels):.2f}, h={y2-y1:d}, w={x2-x1:d}")
    return pred_support_y

def predict(args, support_categorie, modality, support_z_lst, box2seg, propnet, img_array, mask_array, Z, target_size, device, dynamic_crop_ratio, printer,
            norm_a=None, norm_b=None):
    propnet.eval()
    batch = {
        "support_x": [], 
        "support_y": [],
        "query_xs": [],
        "query_slice_ids": [],
        "crop_corners": [],
        "crop_sizes": [],
    }
    for support_z in support_z_lst:
        support_x = img_array[support_z]
        support_y = mask_array[support_z]
        if box2seg is not None:
            support_y = infer_by_box2seg(args, support_categorie, box2seg, support_z, support_x, support_y, dynamic_crop_ratio, device, printer)
            if support_y is None:
                continue
            mask_array[support_z] = support_y
        if support_y.max() == 0: 
            continue
        start = max(support_z - Z, 0)
        end = min(support_z + Z + 1, len(img_array))
        # normalize images according to modality
        if modality == "CT":
            if norm_a is None or norm_b is None:
                norm_a, norm_b = compute_dynamic_window(support_x, support_y)
            window_center = norm_a
            window_width = norm_b
            support_x = normalize(window_transform(support_x, window_center, window_width))
            query_xs = [normalize(window_transform(img_array[i], window_center, window_width)) for i in range(start, end) if i != support_z]
        elif modality in ["MR", "PETCT", "microCT", "SRX", "EM"]:
            if norm_a is None or norm_b is None:
                norm_a, norm_b =  compute_dynamic_internal(support_x, support_y)
            minv = norm_a
            maxv = norm_b
            support_x = normalize(internal_transform(support_x, minv, maxv))
            query_xs = [normalize(internal_transform(img_array[i], minv, maxv)) for i in range(start, end) if i != support_z]
        else:
            raise NotImplementedError(modality)
        query_slice_ids = [i for i in range(start, end) if i != support_z]
        cropped_support_x, cropped_support_y, cropped_query_xs, x1, y1, x2, y2 = center_crop(support_x, support_y, query_xs, dynamic_crop_ratio=dynamic_crop_ratio)
        crop_corners = [[x1, y1, x2, y2] for i in range(start, end) if i != support_z]
        crop_sizes = [[*cropped_query_x.shape] for cropped_query_x in cropped_query_xs]
        support_augmented = transform(image=cropped_support_x, mask=cropped_support_y)
        support_x = support_augmented['image'] # (1, 224, 224)
        support_y = (support_augmented['mask'] > 0.5).float() # (224, 224)
        for i, cropped_query_x in enumerate(cropped_query_xs):
            query_augmented = transform(image=cropped_query_x)
            query_x = query_augmented['image']
            query_xs[i] = query_x 
        if len(query_xs) == 0:
            continue
        query_xs = torch.stack(query_xs, dim=0) # (N_queries, 1, 224, 224)
        batch["support_x"].append(support_x)
        batch["support_y"].append(support_y)
        batch["query_xs"].append(torch.Tensor(query_xs))
        batch["query_slice_ids"].append(torch.Tensor(query_slice_ids))
        batch["crop_corners"].append(torch.Tensor(crop_corners))
        batch["crop_sizes"].append(torch.Tensor(crop_sizes))
    if len(batch["crop_sizes"]) == 0:
        return None, None, None, None, None, None
    for key in batch:
        if key in ["query_xs", "query_slice_ids", "crop_corners", "crop_sizes"]:
            batch[key] = pad_to_match_max_length(batch[key])
        batch[key] = torch.stack(batch[key], dim=0).float().to(device)
    with torch.no_grad():
        pred_dict = propnet(batch)
        preds = [pred_dict["predictions"][0]]
        pred_size = preds[0].size()[-2:]
        for pred in pred_dict["predictions"][1:]:
            preds.append(F.interpolate(pred, size=pred_size, mode="bilinear", align_corners=True))
        preds = torch.mean(torch.stack(preds, dim=0), dim=0)
        preds = torch.argmax(preds, dim=1) # (BNq, H, W)
    B, Nq, _, H, W = batch["query_xs"].shape
    return preds, batch["query_slice_ids"].view(B*Nq), batch["crop_corners"].view(B*Nq, 4), batch["crop_sizes"].view(B*Nq, 2), norm_a, norm_b

def update_mask(mask_array, num_array, preds, query_slice_ids, crop_corners, crop_sizes):
    preds = preds.detach().cpu().numpy()
    query_slice_ids = query_slice_ids.detach().cpu().numpy()
    crop_corners = crop_corners.detach().cpu().numpy()
    crop_sizes = crop_sizes.detach().cpu().numpy()
    for pred, query_slice_id, (x1, y1, x2, y2), (h, w) in zip(preds, query_slice_ids, crop_corners, crop_sizes):
        pred = cv2.resize(pred, (int(w), int(h)), interpolation=cv2.INTER_NEAREST)
        mask_array[int(query_slice_id)][int(y1):int(y2), int(x1):int(x2)] += pred[:int(y2-y1), :int(x2-x1)]
        num_array[int(query_slice_id)] += 1
    return mask_array, num_array

def infer(args, modality, box2seg, propnet, 
            img, img_array, spacing, 
            mask, support_category_ids, support_categories, support_mask_arrays, support_zs,
            save_file, target_size=224, dynamic_crop_ratio=1.5, neighbor_spacing_z=20, max_Z=40, device="cuda:0", printer=print):
    used_times = []
    pred_array = np.zeros_like(support_mask_arrays[0]).astype(support_mask_arrays[0].dtype)
    for support_category_id, support_categorie, support_mask_array, support_z in zip(
        support_category_ids, support_categories, support_mask_arrays, support_zs):
        # if support_categorie not in ["liverbloodvessel",]:
        #     continue
        used_time, _, pred_array_one = infer_one_sample(args, support_categorie, modality, box2seg, propnet, 
                                    img, img_array, spacing, 
                                    mask, support_mask_array, support_z,
                                    save_file, target_size, dynamic_crop_ratio, neighbor_spacing_z, max_Z, device, printer)
        if used_time is None:
            continue
        pred_array[pred_array_one > 0] = support_category_id
        used_times.append(used_time)
    return np.sum(used_times), pred_array

def infer_one_sample(args, support_categorie, modality, box2seg, propnet, 
            img, img_array, spacing, 
            mask, support_mask_array, support_z,
            save_file, target_size=224, dynamic_crop_ratio=1.5, neighbor_spacing_z=20, max_Z=40, device="cuda:0", printer=print):

    spacing = img.GetSpacing()  # x, y, z
    Z = min(int(neighbor_spacing_z / spacing[2]), max_Z)

    mask_array = (support_mask_array > 0).astype(np.int16)  # z, y, x 
    init_mask_array = support_mask_array
    num_array = np.zeros(mask_array.shape[0])
    proped_slice_ids = []

    start_time = time.time()

    num_array[support_z] += 1
    preds, query_slice_ids, crop_corners, crop_sizes, norm_a, norm_b = predict(args, support_categorie, modality, [support_z,], box2seg, propnet, img_array, mask_array/num_array[:, np.newaxis, np.newaxis], Z, target_size, device, dynamic_crop_ratio, printer) 
    if preds is None:
        return None, None, None
    mask_array, num_array = update_mask(mask_array, num_array, preds, query_slice_ids, crop_corners, crop_sizes)
    
    while True:
        proped_slice_ids.extend(query_slice_ids.detach().cpu().numpy().tolist())
        support_z_lst = []
        min_query_slice_id = int(min(proped_slice_ids))
        if min_query_slice_id > 0 and np.sum(mask_array[min_query_slice_id]) > 0:
            support_z_lst.append(min_query_slice_id)
        max_query_slice_id = int(max(proped_slice_ids))
        if max_query_slice_id < mask_array.shape[0]-1 and np.sum(mask_array[max_query_slice_id]) > 0:
            support_z_lst.append(max_query_slice_id)
        if len(support_z_lst) == 0:
            break 
        preds, query_slice_ids, crop_corners, crop_sizes, _, _ = predict(args, support_categorie, modality, support_z_lst, None, propnet, img_array, mask_array/num_array[:, np.newaxis, np.newaxis], Z, target_size, device, dynamic_crop_ratio, printer, norm_a=norm_a, norm_b=norm_b)
        if preds is None:
            break
        mask_array, num_array = update_mask(mask_array, num_array, preds, query_slice_ids, crop_corners, crop_sizes)

    mask_array = ((mask_array / num_array[:, np.newaxis, np.newaxis]) > 0.0).astype(np.uint8)
    #mask_array = mask_array.clip(0, 1)
    lcc_mask_array = remove_false_positive(mask_array, init_mask_array)
    used_time = time.time()-start_time

    mask = sitk.GetImageFromArray(mask_array)
    mask.SetSpacing(spacing)
    # sitk.WriteImage(mask, save_file)

    lcc_mask = sitk.GetImageFromArray(lcc_mask_array)
    lcc_mask.SetSpacing(spacing)
    # sitk.WriteImage(lcc_mask, save_file.replace(".nii.gz", "_lcc.nii.gz"))

    return used_time, lcc_mask, lcc_mask_array

def dice_coefficient(output, target, smooth=1e-6):
    intersection = np.sum(output.astype(np.float32) * target.astype(np.float32))
    return (2. * intersection + smooth) / (np.sum(output) + np.sum(target) + smooth)

def calculate_metrics(pred_array, label_array, label_dicts):
    metric_dict = {}
    for label_id, label_name in label_dicts.items():
        dice = dice_coefficient((pred_array == label_id).astype(np.uint8), (label_array == label_id).astype(np.uint8))
        metric_dict[f"dice/{label_name}"] = dice
    return metric_dict

class BasicDataset(object):
    def __init__(self, data_dir, instance_threshold, printer=print):
        super().__init__()
        self.data_dir = data_dir
        self.instance_threshold = instance_threshold
        self.printer = printer
        self.connect_label = None

    def _read_nii_file(self, file):
        img = sitk.ReadImage(file)
        img_array = sitk.GetArrayFromImage(img)
        spacing = img.GetSpacing()
        return img, img_array, spacing

    def _read_dcm_file(self, dcm_file):
        reader = sitk.ImageFileReader()
        reader.SetFileName(dcm_file)
        reader.LoadPrivateTagsOn()
        image = reader.Execute()
        image_array = sitk.GetArrayFromImage(image)
        return image, image_array

    def _read_dcm_dir(self, dcm_dir):
        reader = sitk.ImageSeriesReader()
        dcm_names = reader.GetGDCMSeriesFileNames(dcm_dir)
        reader.SetFileNames(dcm_names)
        image = reader.Execute()
        image_array = sitk.GetArrayFromImage(image)
        return image, image_array

    def _convert_support_mask(self, mask_array, **kwargs):
        max_area = 0
        max_index = -1
        for index, slice_ in enumerate(mask_array):
            area = np.sum(slice_ > 0)
            if area >= max_area:
                max_area = area
                max_index = index
        support_mask_array = np.zeros_like(mask_array).astype(mask_array.dtype)
        support_mask_array[max_index] = mask_array[max_index]
        return support_mask_array, max_index

    def _read_image_mask(self, index):
        img_file = self.img_files[index]
        mask_file = self.mask_files[index]
        img, img_array, spacing = self._read_nii_file(img_file)
        mask, mask_array, _ = self._read_nii_file(mask_file)
        if img_array.shape != mask_array.shape:
            self.printer(f"!!! Failed to load img={img_file}\tmask={mask_file} [img={img_array.shape}\tmask={mask_array.shape}]")
            return None, None, None, None, None, None, None
        return img_file, mask_file, img, img_array, mask, mask_array, spacing

    def _get_patient_id(self, img_file):
        patient_id = os.path.basename(img_file).split('.')[0]
        return patient_id

    def get_image_mask(self, index, connect_label=None, **kwargs):
        img_file, mask_file, img, img_array, mask, mask_array, spacing = self._read_image_mask(index)
        if img_file is None: return None, None, None, None, None, None, None, None, None, None, None
        patient_id = self._get_patient_id(img_file)
        support_category_ids, support_categories, support_mask_arrays, support_zs = [], [], [], []
        self.printer(f"img={img_file}\tmask={mask_file}")
        for k, mask_id in enumerate(self.labels_dict):
            if (self.connect_label is None) or (self.connect_label[k] == True):
                labeled_mask_array = measure.label((mask_array == mask_id).astype(mask_array.dtype))
            elif self.connect_label[k] == False:
                labeled_mask_array = (mask_array == mask_id).astype(mask_array.dtype)
            else:
                raise NotImplementedError(connect_label)
            labeled_labels = np.unique(labeled_mask_array)
            max_instance = 0
            for label_id in labeled_labels:
                if label_id == 0: continue
                instance_mask_array = (labeled_mask_array == label_id)
                max_instance = max(max_instance, np.sum(instance_mask_array))
            instance_threshold = 0.1 * max_instance
            self.printer(f"\t[{mask_id}]\t{self.labels_dict[mask_id]}\tlabeled labels={len(labeled_labels)-1}")
            for label_id in labeled_labels:
                if label_id == 0: continue
                instance_mask_array = (labeled_mask_array == label_id)
                if np.sum(instance_mask_array) < instance_threshold: continue
                support_mask_array, support_z = self._convert_support_mask(instance_mask_array.astype(mask_array.dtype), **kwargs)
                support_mask_arrays.append(support_mask_array)
                support_zs.append(support_z)
                support_categories.append(self.labels_dict[mask_id])
                support_category_ids.append(mask_id)
        self.printer(f"\tTotal {len(support_categories)} instances")
        return patient_id, img_file, img, img_array, spacing, mask, mask_array, support_category_ids, support_categories, support_mask_arrays, support_zs
    
    def get_modality(self):
        return self.data_dir.split("/")[1]

    def __len__(self):
        return len(self.img_files)

class AbdomenCT1k(BasicDataset):
    def __init__(self, data_dir="datasets/CT/AbdomenCT-1K", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.mask_files = glob(os.path.join(self.data_dir, "Mask", "*.nii.gz"))
        self.img_files = [mask_file.replace("Mask", "Image").replace(".nii.gz", "_0000.nii.gz") for mask_file in self.mask_files]
        self.labels_dict = {1: "liver", 2: "kidneys", 3: "spleen", 4: "pancreas"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _get_patient_id(self, img_file):
        patient_id = os.path.basename(img_file).split('.')[0].split('_')[1]
        return patient_id

class AMOSCT(BasicDataset):
    def __init__(self, data_dir="datasets/CT/AMOS-CT", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.img_files = glob(os.path.join(self.data_dir, "imagesTr", "*.nii.gz")) \
                    + glob(os.path.join(self.data_dir, "imagesVa", "*.nii.gz"))
        self.mask_files = [img_file.replace("images", "labels") for img_file in self.img_files]
        self.labels_dict = {
            1: "spleen", 2: "rightkidney", 3: "leftkidney", 4: "gallbladder", 
            5: "esophagus", 6: "liver", 7: "stomach", 8: "arota", 9: "postcava", 
            10: "pancreas", 11: "rightadrenalgland", 12: "leftadrenalgland", 
            13: "duodenum", 14: "bladder", 15: "prostateoruterus"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _get_patient_id(self, img_file):
        patient_id = os.path.basename(img_file).split('.')[0].split('_')[1]
        return patient_id

class AutoPETCT(BasicDataset):
    def __init__(self, data_dir="datasets/CT/AutoPET", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.img_files = glob(os.path.join(data_dir, "FDG/*/*/CTres.nii.gz"))
        self.mask_files = [img_file.replace("CTres.nii.gz", "SEG.nii.gz") for img_file in self.img_files]
        self.labels_dict = {1: "lesion"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")
    
    def _read_image_mask(self, index):
        mask_file = self.mask_files[index]
        mask, mask_array, _ = self._read_nii_file(mask_file)
        img_file = self.img_files[index]
        img, img_array, _ = self._read_nii_file(img_file)
        spacing = img.GetSpacing()
        return img_file, mask_file, img, img_array, mask, mask_array, spacing

    def __len__(self):
        return len(self.img_files)

    def _get_patient_id(self, img_file):
        patient_id = img_file.split("/")[-3].split("_")[-1]
        return patient_id

    def get_modality(self):
        return "CT"

class AutoPETPETCT(BasicDataset):
    def __init__(self, data_dir="datasets/CT/AutoPET", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        img_files = glob(os.path.join(data_dir, "FDG/*/*/PET.nii.gz"))
        self.img_files = []
        self.mask_files = []
        for img_file in img_files:
            mask_file = img_file.replace("PET.nii.gz", "SEG.nii.gz")
            if os.path.isfile(mask_file):
                self.img_files.append(img_file)
                self.mask_files.append(mask_file)
        self.labels_dict = {1: "lesion"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")
    
    def _read_image_mask(self, index):
        mask_file = self.mask_files[index]
        mask, mask_array, _ = self._read_nii_file(mask_file)
        img_file = self.img_files[index]
        img, img_array, _ = self._read_nii_file(img_file)
        spacing = img.GetSpacing()
        return img_file, mask_file, img, img_array, mask, mask_array, spacing

    def __len__(self):
        return len(self.img_files)

    def _get_patient_id(self, img_file):
        patient_id = img_file.split("/")[-3].split("_")[-1]
        return patient_id

    def get_modality(self):
        return "PETCT"

class COVID19SegChallenge(BasicDataset):
    def __init__(self, data_dir="datasets/CT/COVID-19 Seg. Challenge", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.mask_files = glob(os.path.join(self.data_dir, "Train", "*_seg.nii.gz"))
        self.img_files = [mask_file.replace("_seg.nii.gz", "_ct.nii.gz") for mask_file in self.mask_files]
        self.labels_dict = {1: "covid19infections"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _get_patient_id(self, img_file):
        patient_id = os.path.basename(img_file).split('.')[0].split('_')[0].split('-')[3]
        return patient_id
    
class COVID19CTSeg(BasicDataset):
    def __init__(self, data_dir="datasets/CT/COVID-19-CT-Seg", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.mask_files = glob(os.path.join(self.data_dir, "masks", "*.nii.gz"))
        self.img_files = [mask_file.replace("masks", "images") for mask_file in self.mask_files]
        self.labels_dict = {1: "leftlung", 2: "rightlung", 3: "covid19infections"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _get_patient_id(self, img_file):
        if "coronacases" in img_file:
            patient_id = os.path.basename(img_file).split('.')[0].split('_')[1]
        else:
            strs = os.path.basename(img_file).split('.')[0].split('_')
            patient_id = strs[2]+strs[3]
        return patient_id

class AdrenalACCKi67Seg(BasicDataset):
    def __init__(self, data_dir="datasets/CT/Adrenal-ACC-Ki67-Seg/manifest-1680809675630/Adrenal-ACC-Ki67-Seg/", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        segmentation_folders = []
        for root, dirs, files in os.walk(self.data_dir):
            for dir in dirs:
                if "Segmentation" in dir:
                    full_path = os.path.join(root, dir)
                    segmentation_folders.append(full_path)
        self.segmentation_folders = segmentation_folders
        self.labels_dict = {255: "adrenocorticalcarcinoma"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _get_patient_id(self, img_file):
        patient_id = img_file.split("/")[-3].split("_")[-1]
        return patient_id
    
    def _read_image_mask(self, index):
        segmentation_folder = self.segmentation_folders[index]
        mask_file = glob(os.path.join(segmentation_folder, "*.dcm"))
        assert len(mask_file) == 1, segmentation_folder
        mask_file = mask_file[0]
        mask, mask_array = self._read_dcm_file(mask_file)
        root_dir = segmentation_folder.replace(os.path.basename(segmentation_folder), "")
        for folder in os.listdir(root_dir):
            if "Segmentation" in folder:
                continue
            img_file = os.path.join(root_dir, folder)
            img, img_array = self._read_dcm_dir(img_file)
            if img_array.shape[0] == mask_array.shape[0]:
                break
        if img_array.shape[0] != mask_array.shape[0]:
            self.printer(f"!!! Failed to load img={img_file}\tmask={mask_file} [img={img_array.shape}\tmask={mask_array.shape}]")
            return None, None, None, None, None, None, None
        spacing = img.GetSpacing()
        return img_file, mask_file, img, img_array, mask, mask_array, spacing

    def __len__(self):
        return len(self.segmentation_folders)

class CHAOSCT(BasicDataset):
    def __init__(self, data_dir="datasets/CT/CHAOS-CT", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.raw_data_dirs = glob(os.path.join(self.data_dir, "Train_Sets/CT/*/DICOM_anon/"))
        self.labels_dict = {1: "liver"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _read_mask_dir(self, mask_dir, num_slices):
        mask_array = []
        for slice_id in range(num_slices):
            slice_mask = np.asarray(Image.open(os.path.join(mask_dir, f"liver_GT_{slice_id:03d}.png")).convert("L"))
            mask_array.append(slice_mask)
        mask_array = (np.asarray(mask_array) > 0).astype(np.uint8)[::-1]
        return mask_array
    
    def _read_image_mask(self, index):
        raw_data_dir = self.raw_data_dirs[index]
        img, img_array = self._read_dcm_dir(raw_data_dir)
        spacing = img.GetSpacing()
        mask_dir = raw_data_dir.replace("DICOM_anon", "Ground")
        mask_array = self._read_mask_dir(mask_dir, img_array.shape[0])

        if img_array.shape[0] != mask_array.shape[0]:
            self.printer(f"!!! Failed to load img={img_file}\tmask={mask_file} [img={img_array.shape}\tmask={mask_array.shape}]")
            return None, None, None, None, None, None, None
        spacing = img.GetSpacing()
        return raw_data_dir, mask_dir, img, img_array, None, mask_array, spacing

    def __len__(self):
        return len(self.raw_data_dirs)

    def _get_patient_id(self, img_file):
        patient_id = img_file.split("/")[-3]
        return patient_id

class HCCTACESeg(BasicDataset):
    def __init__(self, data_dir="datasets/CT/HCC-TACE-Seg", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.img_files = glob(os.path.join(self.data_dir, "preprocessed/IMG/*_IMG.nii.gz"))
        self.mask_files = [img_file.replace("IMG", "SEG") for img_file in self.img_files]
        self.labels_dict = {1: "liver", 2: "livertumor", 3: "liverbloodvessel"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _get_patient_id(self, img_file):
        patient_id = os.path.basename(img_file).split("_")[1]
        return patient_id

class HECKTOR(BasicDataset):
    def __init__(self, data_dir="datasets/CT/HECKTOR", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.img_files = glob(os.path.join(self.data_dir, "hecktor2022_training/hecktor2022/imagesTr/*_CT.nii.gz"))
        self.mask_files = [img_file.replace("imagesTr", "labelsTr").replace("__CT.nii.gz", ".nii.gz") for img_file in self.img_files]
        self.labels_dict = {1: "headneckprimarytumor", 2: "headnecklymphnodes"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _get_patient_id(self, img_file):
        patient_id = os.path.basename(img_file).split(".")[0].split("-")[1]
        return patient_id

    def get_modality(self):
        return "PETCT"

class INSTANCE(BasicDataset):
    def __init__(self, data_dir="datasets/CT/INSTANCE", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.mask_files = glob(os.path.join(self.data_dir, "train/label/*.nii.gz"))
        #self.mask_files = [file for file in self.mask_files if file in ["datasets/CT/INSTANCE/train/label/094.nii.gz", "datasets/CT/INSTANCE/train/label/001.nii.gz"]]
        self.img_files = [mask_file.replace("label", "data") for mask_file in self.mask_files]
        self.labels_dict = {1: "hematoma"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

class LNQ2023(BasicDataset):
    def __init__(self, data_dir="datasets/CT/LNQ2023", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.mask_files = glob(os.path.join(self.data_dir, "labelsTr/*.nii.gz"))
        self.img_files = [mask_file.replace("labelsTr", "imagesTr").replace(".nii.gz", "_0000.nii.gz") for mask_file in self.mask_files]
        self.labels_dict = {1: "lnq"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

class KiPA(BasicDataset):
    def __init__(self, data_dir="datasets/CT/KiPA", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.mask_files = glob(os.path.join(self.data_dir, "label/*.nii.gz"))
        self.img_files = [mask_file.replace("label", "image") for mask_file in self.mask_files]
        self.labels_dict = {1: "renalvein", 2: "kidney", 3: "renalartery", 4: "kidneytumor"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

class KiTS(BasicDataset):
    def __init__(self, data_dir="datasets/CT/KiTS", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.mask_files = glob(os.path.join(self.data_dir, "dataset/case_*/segmentation.nii.gz"))
        self.img_files = [mask_file.replace("segmentation", "imaging") for mask_file in self.mask_files]
        self.labels_dict = {1: "kidney", 2: "kidneytumor", 3: "kidneycyst"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _read_image_mask(self, index):
        img_file = self.img_files[index]
        mask_file = self.mask_files[index]
        img, img_array, spacing = self._read_nii_file(img_file)
        img_array = np.transpose(img_array, [2, 1, 0])
        spacing = spacing[::-1]
        mask, mask_array, _ = self._read_nii_file(mask_file)
        mask_array = np.transpose(mask_array, [2, 1, 0])
        if img_array.shape != mask_array.shape:
            self.printer(f"!!! Failed to load img={img_file}\tmask={mask_file} [img={img_array.shape}\tmask={mask_array.shape}]")
            return None, None, None, None, None, None, None
        return img_file, mask_file, img, img_array, mask, mask_array, spacing

    def _get_patient_id(self, img_file):
        patient_id = img_file.split("/")[-2].split("_")[1]
        return patient_id

class LymphNodes(BasicDataset):
    def __init__(self, data_dir="datasets/CT/Lymph Nodes", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.img_files = glob(os.path.join(self.data_dir, "preprocessed/IMG/*_IMG.nii.gz"))
        self.mask_files = [img_file.replace("_IMG.nii.gz", "_mask.nii.gz").replace("IMG", "SEG") for img_file in self.img_files]
        self.labels_dict = {i: "mediastinallymphnodes" for i in range(1, 2001)}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _read_image_mask(self, index):
        img_file = self.img_files[index]
        mask_file = self.mask_files[index]
        img, img_array, spacing = self._read_nii_file(img_file)
        mask, mask_array, _ = self._read_nii_file(mask_file)
        mask_array = mask_array.astype(np.uint8)
        self.labels_dict = {i: "mediastinallymphnodes" for i in range(1, np.max(mask_array)+1)}
        return img_file, mask_file, img, img_array, mask, mask_array, spacing

    def _get_patient_id(self, img_file):
        patient_id = os.path.basename(img_file).split("_")[2]
        return patient_id

class Task03Liver(BasicDataset):
    def __init__(self, data_dir="datasets/CT/Task03_Liver", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.mask_files = glob(os.path.join(self.data_dir, "labelsTr/*.nii.gz"))
        self.img_files = [mask_file.replace("labelsTr", "imagesTr") for mask_file in self.mask_files]
        self.labels_dict = {1: "liver", 2: "livercancer"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _get_patient_id(self, img_file):
        patient_id = os.path.basename(img_file).split('.')[0].split('_')[1]
        return patient_id

class Task06Lung(Task03Liver):
    def __init__(self, data_dir="datasets/CT/Task06_Lung", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.labels_dict = {1: "lungcancer"}

class Task07Pancreas(Task03Liver):
    def __init__(self, data_dir="datasets/CT/Task07_Pancreas", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.labels_dict = {1: "pancreas", 2: "pancreascancer"}

class Task08HepaticVessel(Task03Liver):
    def __init__(self, data_dir="datasets/CT/Task08_HepaticVessel", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.labels_dict = {1: "hepaticvessel", 2: "hepatictumour"}

class Task09Spleen(Task03Liver):
    def __init__(self, data_dir="datasets/CT/Task09_Spleen", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.labels_dict = {1: "spleen"}

class Task10Colon(Task03Liver):
    def __init__(self, data_dir="datasets/CT/Task10_Colon", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.labels_dict = {1: "coloncancerprimaries"}

class NSCLCPleuralEffusion(BasicDataset):
    def __init__(self, data_dir="datasets/CT/NSCLC Pleural Effusion", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.mask_files = glob(os.path.join(self.data_dir, "*/*/*_reviewer.nii.gz"))
        self.raw_dir = os.path.join(self.data_dir, "raw_data/manifest-1586193031612/NSCLC-Radiomics/")
        self.printer(f"[{data_dir}] {self.__len__()} samples")
    
    def _read_image_mask(self, index):
        mask_file = self.mask_files[index]
        mask_name = mask_file.split("/")[-3].lower().replace("_", "")
        mask, mask_array, _ = self._read_nii_file(mask_file)
        mask_array = mask_array[:, ::-1]
        # self.labels_dict = {i: "pleuraleffusion" for i in range(1, np.max(mask_array)+1)}
        self.labels_dict = {i: mask_name for i in range(1, np.max(mask_array)+1)}
        img_dir = glob(os.path.join(self.raw_dir, mask_file.split("/")[-2], "*", "*"))
        if len(img_dir) != 1:
            self.printer(f"!!! Failed to load img={img_dir} [{len(img_dir)} len(img_dir) != 1]")
            return None, None, None, None, None, None, None
        img_dir = img_dir[0]
        img, img_array = self._read_dcm_dir(img_dir)
        spacing = img.GetSpacing()
        return img_dir, mask_file, img, img_array, mask, mask_array, spacing

    def __len__(self):
        return len(self.mask_files)

    def _get_patient_id(self, img_file):
        patient_id = img_file.split("/")[-3]
        return patient_id

class TotalSegmentator(BasicDataset):
    def __init__(self, data_dir="datasets/CT/TotalSegmentator", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.mask_files = glob(os.path.join(self.data_dir, "*/segmentations/*.nii.gz"))
        self.img_files =  [mask_file.replace(os.path.basename(mask_file), "").replace("segmentations/", "ct.nii.gz") for mask_file in self.mask_files]
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _read_nii_file(self, file):
        try:
            img = sitk.ReadImage(file)
            img_array = sitk.GetArrayFromImage(img)
            spacing = img.GetSpacing()
            return img, img_array, spacing
        except Exception as e:
            self.printer(e)
            return None, None, None

    def _read_image_mask(self, index):
        img_file = self.img_files[index]
        mask_file = self.mask_files[index]
        img, img_array, spacing = self._read_nii_file(img_file)
        if img is None:
            return None, None, None, None, None, None, None
        mask, mask_array, _ = self._read_nii_file(mask_file)
        mask_name = os.path.basename(mask_file).replace(".nii.gz", "").replace("_", "").lower()
        self.labels_dict = {i: mask_name for i in range(1, np.max(mask_array)+1)}
        return img_file, mask_file, img, img_array, mask, mask_array, spacing

    def _get_patient_id(self, img_file):
        patient_id = img_file.split('/')[-2]
        return patient_id

class WORD(BasicDataset):
    def __init__(self, data_dir="datasets/CT/WORD", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir
        self.img_files = glob(os.path.join(self.data_dir, "WORD-V0.1.0/imagesTr/*.nii.gz")) \
                    + glob(os.path.join(self.data_dir, "WORD-V0.1.0/imagesVal/*.nii.gz"))
        self.mask_files =  [img_file.replace("images", "labels") for img_file in self.img_files]
        self.labels_dict = {1: "liver", 2: "spleen", 3: "left_kidney", 4: "right_kidney",
                            5: "stomach", 6: "gallbladder", 7: "esophagus", 8: "pancreas",
                            9: "duodenum", 10: "colon", 11: "intestine", 12: "adrenal",
                            13: "rectum", 14: "bladder", 15: "Head_of_femur_L", 16: "Head_of_femur_R"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _get_patient_id(self, img_file):
        patient_id = os.path.basename(img_file).split('.')[0].split('_')[1]
        return patient_id

class HaNSegCT(BasicDataset):
    def __init__(self, data_dir="datasets/CT/HaN-Seg-CT", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.mask_files = glob(os.path.join(self.data_dir, "preprocessed/SEG/*/*.seg.nrrd"))
        self.img_files = [mask_file.replace("SEG", "IMG") for mask_file in self.mask_files]
        self.img_files = [img_file.replace(os.path.basename(img_file), "_".join([*os.path.basename(img_file).split("_")[:2], "IMG", "CT"])+".nrrd") for img_file in self.img_files]
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _read_image_mask(self, index):
        img_file = self.img_files[index]
        mask_file = self.mask_files[index]
        img, img_array, spacing = self._read_nii_file(img_file)
        mask, mask_array, _ = self._read_nii_file(mask_file)
        mask_name = "".join(os.path.basename(mask_file).split(".")[0].split("_")[3:]).lower()
        self.labels_dict = {i: mask_name for i in np.unique(mask_array) if i != 0}
        return img_file, mask_file, img, img_array, mask, mask_array, spacing

    def _get_patient_id(self, img_file):
        patient_id = os.path.basename(img_file).split('.')[0].split('_')[1]
        return patient_id

class QUBIQCT(BasicDataset):
    def __init__(self, data_dir="datasets/CT/QUBIQ-CT", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.mask_files = glob(os.path.join(self.data_dir, "Training/*/*/*seg*.nii.gz")) \
                        + glob(os.path.join(self.data_dir, "Validation/*/Validation/*/*seg*.nii.gz"))
        self.img_files =[mask_file.replace(os.path.basename(mask_file), "image.nii.gz") for mask_file in self.mask_files]
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _read_image_mask(self, index):
        img_file = self.img_files[index]
        mask_file = self.mask_files[index]
        img, img_array, spacing = self._read_nii_file(img_file)
        if len(img_array.shape) < 3:
            self.printer(f"!!! Pass {img_file} is 2D image ({img_array.shape})")
            return None, None, None, None, None, None, None
        img_array = np.transpose(img_array, (2, 1, 0))
        spacing = spacing[::-1]
        mask, mask_array, _ = self._read_nii_file(mask_file)
        mask_array = mask_array.astype(np.uint8)
        mask_array = np.transpose(mask_array, (2, 1, 0))
        if "Validation" in img_file: mask_name = img_file.split("/")[-4].replace("-", "")
        else: mask_name = img_file.split("/")[-3].replace("-", "")
        self.labels_dict = {i: mask_name for i in np.unique(mask_array) if i != 0}
        return img_file, mask_file, img, img_array, mask, mask_array, spacing

    def _get_patient_id(self, img_file):
        patient_id = img_file.split("/")[-2].replace("-", "")
        return patient_id

class ACDC(BasicDataset):
    def __init__(self, data_dir="datasets/MR/ACDC", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.mask_files = glob(os.path.join(self.data_dir, "training/*/*_gt.nii.gz")) \
                        + glob(os.path.join(self.data_dir, "testing/*/*_gt.nii.gz"))
        self.img_files =  [mask_file.replace("_gt.nii.gz", ".nii.gz") for mask_file in self.mask_files]
        self.labels_dict = {1: "rightventricle", 2: "myocardium", 3: "leftventricle"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _get_patient_id(self, img_file):
        patient_id = os.path.basename(img_file).split('_')[0]
        return patient_id

class AMOSMR(BasicDataset):
    def __init__(self, data_dir="datasets/MR/AMOS-MR", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.img_files = glob(os.path.join(self.data_dir, "imagesTr/*.nii.gz")) \
                    + glob(os.path.join(self.data_dir, "imagesVa/*.nii.gz"))
        self.mask_files =  [img_file.replace("images", "labels") for img_file in self.img_files]
        self.labels_dict = {1: "spleen", 2: "rightkidney", 3: "leftkidney", 
                            4: "gallbladder", 5: "esophagus", 6: "liver", 7: "stomach", 
                            8: "arota", 9: "postcava", 10: "pancreas", 11: "rightadrenalgland", 
                            12: "leftadrenalgland", 13: "duodenum", 14: "bladder", 15: "prostateoruterus"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _get_patient_id(self, img_file):
        patient_id = os.path.basename(img_file).split('.')[0].split('_')[1]
        return patient_id

class ATLASR20(BasicDataset):
    def __init__(self, data_dir="datasets/MR/ATLAS-R2.0", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.mask_files = glob(os.path.join(self.data_dir, "Training/*/*/*/*/*_mask.nii.gz"))
        self.img_files = []
        for mask_file in self.mask_files:
            img_files = glob(os.path.join(mask_file.replace(os.path.basename(mask_file), ""), "*_T1w.nii.gz"))
            assert len(img_files) == 1, img_files
            self.img_files.append(img_files[0])
        self.labels_dict = {1: "brainstroke"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _get_patient_id(self, img_file):
        patient_id = os.path.basename(img_file).split('-')[1].split("_")[0]
        return patient_id

class BraTS(BasicDataset):
    def __init__(self, data_dir="datasets/MR/BraTS", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.mask_files = []
        self.img_files = []
        for mask_file in glob(os.path.join(self.data_dir, "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/*/*-seg.nii.gz")):
            for img_file in glob(os.path.join(mask_file.replace(os.path.basename(mask_file), ""), "*.nii.gz")):
                if img_file.endswith("-seg.nii.gz"):
                    continue
                self.img_files.append(img_file)
                self.mask_files.append(mask_file)
        self.labels_dict = {1: "braintumornonenhancingtumorcore",
                            2: "braintumorperitumoraledema",
                            3: "braintumorenhancingtumor"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _get_patient_id(self, img_file):
        patient_id = "".join(os.path.basename(img_file).split("-")[2:4])
        return patient_id

class CCTumorHeterogeneity(BasicDataset):
    def __init__(self, data_dir="datasets/MR/CC-Tumor-Heterogeneity", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.mask_files = glob(os.path.join(self.data_dir, "preprocessed/*/*/mask_*.nii.gz"))
        self.img_files = []
        for mask_file in self.mask_files:
            img_files = glob(os.path.join(mask_file.replace(os.path.basename(mask_file), ""), "image.nii.gz"))
            assert len(img_files) == 1, mask_file
            self.img_files.append(img_files[0])
        self.labels_dict = {255: "cervicalcancer"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _read_image_mask(self, index):
        img_file = self.img_files[index]
        mask_file = self.mask_files[index]
        img, img_array, spacing = self._read_nii_file(img_file)
        mask, mask_array, _ = self._read_nii_file(mask_file)
        ind = np.where(mask_array)
        print(spacing, mask_array.shape, set(ind[0]))
        return None, None, None, None, None, None, None
        if img_array.shape != mask_array.shape:
            self.printer(f"!!! Failed to load img={img_file}\tmask={mask_file} [img={img_array.shape}\tmask={mask_array.shape}]")
            return None, None, None, None, None, None, None
        return img_file, mask_file, img, img_array, mask, mask_array, spacing


    def _get_patient_id(self, img_file):
        patient_id = img_file.split("/")[-3].split("-")[1]
        return patient_id

class CHAOSMR(BasicDataset):
    def __init__(self, data_dir="datasets/MR/CHAOS-MR", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.img_dirs = glob(os.path.join(self.data_dir, "Train_Sets/MR/*/T1DUAL/DICOM_anon/InPhase/")) \
            + glob(os.path.join(self.data_dir, "/Train_Sets/MR/*/T2SPIR/DICOM_anon/"))
        self.mask_dirs = []
        for img_dir in self.img_dirs:
            if "T1DUAL" in img_dir:
                mask_dir = img_dir.replace("InPhase/", "").replace("OutPhase/", "").replace("DICOM_anon", "Ground")
            else:
                mask_dir = img_dir.replace("DICOM_anon", "Ground")
            self.mask_dirs.append(mask_dir)
        self.labels_dict = {63: "liver", 126: "leftkidney", 189: "rightkidney", 252: "spleen"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _read_mask_dir(self, mask_dir, num_slices):
        mask_array = []
        ind = os.path.basename(glob(os.path.join(mask_dir, "IMG-*.png"))[0]).split("-")[1]
        if "T1DUAL" in mask_dir:
            for slice_id in range(num_slices):
                slice_mask = np.asarray(Image.open(os.path.join(mask_dir, f"IMG-{ind}-{(slice_id+1)*2:05d}.png")).convert("L"))
                mask_array.append(slice_mask)
        else:
            for slice_id in range(num_slices):
                slice_mask = np.asarray(Image.open(os.path.join(mask_dir, f"IMG-{ind}-{slice_id+1:05d}.png")).convert("L"))
                mask_array.append(slice_mask)
        mask_array = np.asarray(mask_array).astype(np.uint8)
        return mask_array

    def _read_image_mask(self, index):
        img_dir = self.img_dirs[index]
        mask_dir = self.mask_dirs[index]
        img, img_array = self._read_dcm_dir(img_dir)
        mask_array = self._read_mask_dir(mask_dir, img_array.shape[0])
        spacing = img.GetSpacing()
        return img_dir, mask_dir, img, img_array, None, mask_array, spacing

    def _get_patient_id(self, img_file):
        if "T1DUAL" in img_file:
            strs = img_file.split("/")
            patient_id = strs[-5]+strs[-1]
        else:
            strs = img_file.split("/")
            patient_id = strs[-4]
        return patient_id

    def __len__(self):
        return len(self.img_dirs)

class ISLES(BasicDataset):
    def __init__(self, data_dir="datasets/MR/ISLES", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.img_files = glob(os.path.join(self.data_dir, "ISLES-2022/sub-strokecase*/*/dwi/*.nii.gz"))
        self.mask_files = []
        for img_file in self.img_files:
            strs = img_file.split('/')
            mask_files = glob(os.path.join(img_file.replace("ISLES-2022/", "ISLES-2022/derivatives/").replace(strs[-2]+"/", "").replace(strs[-1], ""), "*_msk.nii.gz"))
            assert len(mask_files) == 1, img_file
            self.mask_files.append(mask_files[0])
        self.labels_dict = {1: "ischemicstrokelesion"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _get_patient_id(self, img_file):
        strs = img_file.split('/')
        patient_id = strs[-4][-4:]
        return patient_id

class MnM2(BasicDataset):
    def __init__(self, data_dir="datasets/MR/MnM2", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.mask_files = glob(os.path.join(self.data_dir, "dataset/*/*_gt.nii.gz"))
        self.img_files = [mask_file.replace("_gt.nii.gz", ".nii.gz") for mask_file in self.mask_files]
        self.labels_dict = {1: "leftventricle", 2: "myocardium", 3: "rightventricle"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _read_image_mask(self, index):
        img_file = self.img_files[index]
        mask_file = self.mask_files[index]
        try:
            img, img_array, spacing = self._read_nii_file(img_file)
        except:
            self.printer(f"!!! ITK ERROR: ITK only supports orthonormal direction cosines.  No orthonormal definition found! img_file={img_file}")
            return None, None, None, None, None, None, None
        if img_array.shape[0] == 1:
            self.printer(f"!!! It is a 2D image with shape of {img_array.shape} img_file={img_file}")
            return None, None, None, None, None, None, None
        mask, mask_array, _ = self._read_nii_file(mask_file)
        if img_array.shape != mask_array.shape:
            self.printer(f"!!! Failed to load img={img_file}\tmask={mask_file} [img={img_array.shape}\tmask={mask_array.shape}]")
            return None, None, None, None, None, None, None
        return img_file, mask_file, img, img_array, mask, mask_array, spacing

    def _get_patient_id(self, img_file):
        patient_id = img_file.split('/')[-2]
        return patient_id


class Task01BrainTumour(Task03Liver):
    def __init__(self, data_dir="datasets/MR/Task01_BrainTumour", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.mask_files = []
        for modality_id in [0, 1, 2, 3]:
            self.mask_files.extend([mask_file+f".{modality_id}" for mask_file in glob(os.path.join(self.data_dir, "labelsTr/*.nii.gz"))])
        self.img_files = [mask_file.replace("labelsTr", "imagesTr") for mask_file in self.mask_files]
        self.labels_dict = {1: "edema", 2: "nonenhancingtumor", 3: "enhancingtumor"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")
    
    def _read_image_mask(self, index):
        img_file = self.img_files[index]
        mask_file = self.mask_files[index]
        modality_id = int(mask_file.split(".")[-1])
        img_file = img_file[:-2]
        mask_file = mask_file[:-2]
        img, img_array, spacing = self._read_nii_file(img_file)
        img_array = img_array[modality_id]
        mask, mask_array, _ = self._read_nii_file(mask_file)
        if img_array.shape != mask_array.shape:
            self.printer(f"!!! Failed to load img={img_file}\tmask={mask_file} [img={img_array.shape}\tmask={mask_array.shape}]")
            return None, None, None, None, None, None, None
        return img_file, mask_file, img, img_array, mask, mask_array, spacing

class Task02Heart(Task03Liver):
    def __init__(self, data_dir="datasets/MR/Task02_Heart", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.labels_dict = {1: "leftatrium"}

class Task04Hippocampus(Task03Liver):
    def __init__(self, data_dir="datasets/MR/Task04_Hippocampus", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.labels_dict = {1: "anterior", 2: "posterior"}

class Task05Prostate(Task03Liver):
    def __init__(self, data_dir="datasets/MR/Task05_Prostate", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.mask_files = []
        for modality_id in [0, 1]:
            self.mask_files.extend([mask_file+f".{modality_id}" for mask_file in glob(os.path.join(self.data_dir, "labelsTr/*.nii.gz"))])
        self.img_files = [mask_file.replace("labelsTr", "imagesTr") for mask_file in self.mask_files]
        self.labels_dict = {1: "PZ", 2: "TZ"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")
    
    def _read_image_mask(self, index):
        img_file = self.img_files[index]
        mask_file = self.mask_files[index]
        modality_id = int(mask_file.split(".")[-1])
        img_file = img_file[:-2]
        mask_file = mask_file[:-2]
        img, img_array, spacing = self._read_nii_file(img_file)
        img_array = img_array[modality_id]
        mask, mask_array, _ = self._read_nii_file(mask_file)
        return img_file, mask_file, img, img_array, mask, mask_array, spacing

class Spine(BasicDataset):
    def __init__(self, data_dir="datasets/MR/Spine", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        mask_files = glob(os.path.join(self.data_dir, "SpineSegmented/*/L*.mha")) \
                        + glob(os.path.join(self.data_dir, "SpineSegmented/*/S*.mha")) \
                        + glob(os.path.join(self.data_dir, "SpineSegmented/*/T*.mha"))
        self.mask_files, self.img_files = [], []
        for mask_file in mask_files:
            img_file = os.path.join(mask_file.replace("SpineSegmented", "SpineDatasets").replace(os.path.basename(mask_file), "")[:-1]+".dcm")
            if not os.path.exists(img_file):
                continue
            self.mask_files.append(mask_file)
            self.img_files.append(img_file)
        self.labels_dict = {1: "spine"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _read_image_mask(self, index):
        img_file = self.img_files[index]
        mask_file = self.mask_files[index]
        img, img_array = self._read_dcm_file(img_file)
        mask, mask_array, _ = self._read_nii_file(mask_file)
        mask_array = (mask_array > 0).astype(np.uint8)
        spacing = img.GetSpacing()
        return img_file, mask_file, img, img_array, mask, mask_array, spacing

    def _get_patient_id(self, img_file):
        patient_id = os.path.basename(img_file).split(".")[0]
        return patient_id

class QinProstateRepeatability(BasicDataset):
    def __init__(self, data_dir="datasets/MR/Qin-Prostate-Repeatability", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.mask_files = glob(os.path.join(self.data_dir, "preprocessed/SEG/*/*_Segmentations_*.nii.gz"))
        self.img_files = [mask_file.replace("SEG", "IMG").replace("_Segmentations", "").replace("_1", "").replace("_2", "").replace("_3", "").replace("_4", "") for mask_file in self.mask_files]
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _read_image_mask(self, index):
        img_file = self.img_files[index]
        mask_file = self.mask_files[index]
        img, img_array, spacing = self._read_nii_file(img_file)
        mask, mask_array, _ = self._read_nii_file(mask_file)
        if img_array.shape != mask_array.shape:
            self.printer(f"!!! Failed to load img={img_file}\tmask={mask_file} [img={img_array.shape}\tmask={mask_array.shape}]")
            return None, None, None, None, None, None, None
        if "Segmentations_1" in mask_file: mask_name = "prostategland"
        elif "Segmentations_2" in mask_file: mask_name = "prostateglandperipheralzone"
        elif "Segmentations_3" in mask_file: mask_name = "prostatesuspectedtumor"
        elif "Segmentations_4" in mask_file: mask_name = "prostate"
        else: raise NotImplementedError
        self.labels_dict = {i: mask_name for i in range(1, np.max(mask_array)+1)}
        return img_file, mask_file, img, img_array, mask, mask_array, spacing

    def _get_patient_id(self, img_file):
        patient_id = img_file.split("/")[-2].split("-")[1]
        return patient_id

class PROMISE(BasicDataset):
    def __init__(self, data_dir="datasets/MR/PROMISE", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.mask_files = glob(os.path.join(self.data_dir, "livechallenge_test_data/*_segmentation.mhd")) \
                    + glob(os.path.join(self.data_dir, "test_data/*_segmentation.mhd")) \
                    + glob(os.path.join(self.data_dir, "training_data/*_segmentation.mhd"))
        self.img_files = [mask_file.replace("_segmentation", "") for mask_file in self.mask_files]
        printer(f"Total files: {len(self.img_files)}")
        self.labels_dict = {1: "prostate"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _get_patient_id(self, img_file):
        patient_id = os.path.basename(img_file).split(".")[0]
        return patient_id

class PICAI(BasicDataset):
    def __init__(self, data_dir="datasets/MR/PI-CAI", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        mask_files = glob(os.path.join(self.data_dir, "pical_labels/csPCa_lesion_delineations/AI/Bosma22a/*.nii.gz")) \
            + glob(os.path.join(self.data_dir, "pical_labels/csPCa_lesion_delineations/human_expert/resampled/*.nii.gz"))
        self.img_files = []
        self.mask_files = []
        for mask_file in mask_files:
            strs = os.path.basename(mask_file).replace(".nii.gz", "").split("_")
            img_files = glob(os.path.join(self.data_dir, "picai_*", strs[0], f"{strs[0]}_{strs[1]}_*.mha"))
            for img_file in img_files:
                self.img_files.append(img_file)
                self.mask_files.append(mask_file)
        self.labels_dict = {1: "prostatecancer"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _read_image_mask(self, index):
        img_file = self.img_files[index]
        mask_file = self.mask_files[index]
        img, img_array, spacing = self._read_nii_file(img_file)
        mask, mask_array, _ = self._read_nii_file(mask_file)
        mask_array = (mask_array >= 1).astype(np.uint8)
        if img_array.shape != mask_array.shape:
            self.printer(f"!!! Failed to load img={img_file}\tmask={mask_file} [img={img_array.shape}\tmask={mask_array.shape}]")
            return None, None, None, None, None, None, None
        return img_file, mask_file, img, img_array, mask, mask_array, spacing

    def _get_patient_id(self, img_file):
        patient_id = img_file.split("/")[-2]
        return patient_id

class WMH(BasicDataset):
    def __init__(self, data_dir="datasets/MR/WMH", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        mask_files = glob(os.path.join(self.data_dir, "training/Amsterdam/GE3T/*/wmh.nii.gz")) \
                        + glob(os.path.join(self.data_dir, "training/Singapore/*/wmh.nii.gz")) \
                        + glob(os.path.join(self.data_dir, "training/Utrecht/*/wmh.nii.gz")) \
                        + glob(os.path.join(self.data_dir, "test/Amsterdam/GE1T5/*/wmh.nii.gz")) \
                        + glob(os.path.join(self.data_dir, "test/Amsterdam/GE3T/*/wmh.nii.gz")) \
                        + glob(os.path.join(self.data_dir, "test/Amsterdam/Philips_VU .PETMR_01./*/wmh.nii.gz")) \
                        + glob(os.path.join(self.data_dir, "test/Singapore/*/wmh.nii.gz")) \
                        + glob(os.path.join(self.data_dir, "test/Utrecht/*/wmh.nii.gz"))
        self.mask_files, self.img_files = [], []
        for mask_file in mask_files:
            img_dir = mask_file.replace("wmh.nii.gz", "pre")
            for img_file in glob(os.path.join(img_dir, "*.nii.gz")):
                self.mask_files.append(mask_file)
                self.img_files.append(img_file)
        self.labels_dict = {1: "whitematterhyperintensities", 2: "wmhotherpathology"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _get_patient_id(self, img_file):
        patient_id = img_file.split("/")[-3]
        return patient_id

class NCIISBI(BasicDataset):
    def __init__(self, data_dir="datasets/MR/NCI-ISBI", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.img_files = []
        self.mask_files = []
        mask_files = glob(os.path.join(self.data_dir, "Mask/*.nrrd"))
        for mask_file in mask_files:
            img_dcm_dir = glob(os.path.join(mask_file.replace("Mask", "Image").replace(".nrrd", "").replace("_truth", ""), "*", "*"))
            if len(img_dcm_dir) != 1: continue
            img_dcm_dir = img_dcm_dir[0]
            self.img_files.append(img_dcm_dir)
            self.mask_files.append(mask_file)
        self.labels_dict = {1: "prostateperipheral", 2: "prostatecentralgland"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _read_image_mask(self, index):
        img_file = self.img_files[index]
        mask_file = self.mask_files[index]
        img, img_array = self._read_dcm_dir(img_file)
        mask, mask_array, _ = self._read_nii_file(mask_file)
        if img_array.shape != mask_array.shape:
            self.printer(f"!!! Failed to load img={img_file}\tmask={mask_file} [img={img_array.shape}\tmask={mask_array.shape}]")
            return None, None, None, None, None, None, None
        spacing = img.GetSpacing()
        return img_file, mask_file, img, img_array, mask, mask_array, spacing

    def _get_patient_id(self, img_file):
        patient_id = img_file.split("/")[-3]
        return patient_id

class HM3dSeg(BasicDataset):
    def __init__(self, data_dir="infer_data/HM-3d/final_project_image/", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.mask_files = glob(os.path.join(self.data_dir, "final_project_mask", "*.nii.gz"))
        self.img_files = [mask_file.replace("_mask", "_image") for mask_file in self.mask_files]
        self.labels_dict = {1: "muscle", 2: "extravisceralfat", 3: "visceralfat"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _get_patient_id(self, img_file):
        patient_id = os.path.basename(img_file).split('_')[0]
        return patient_id

    def get_modality(self):
        return "CT"

class TCGASTAD(BasicDataset):
    def __init__(self, data_dir="infer_data/TCGA-STAD", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.mask_files = glob(os.path.join(self.data_dir, "labels", "*.nii.gz"))
        self.img_files = [mask_file.replace("labels/", "images/") for mask_file in self.mask_files]
        self.labels_dict = {1: "gastriccancer"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _get_patient_id(self, img_file):
        patient_id = os.path.basename(img_file).split('.')[0]
        return patient_id

    def get_modality(self):
        return "CT"


class GCSegPKCancer(BasicDataset):
    def __init__(self, data_dir="datasets/CT/GCSeg_PKCancer", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.mask_files = glob(os.path.join(self.data_dir, "labels/*.nii.gz"))
        self.img_files = [mask_file.replace("labels", "images").replace(".nii.gz", "_0000.nii.gz") for mask_file in self.mask_files]
        self.labels_dict = {1: "gastriccancer"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _get_patient_id(self, img_file):
        patient_id = os.path.basename(img_file).split("_")[1]
        return patient_id

class MouseKidneySRX(BasicDataset):
    def __init__(self, data_dir="datasets/SRX/MouseKidneySRX", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.mask_files = glob(os.path.join(self.data_dir, "seg*.tif"))
        self.img_files = [mask_file.replace("seg", "image") for mask_file in self.mask_files]
        self.labels_dict = {1: "Glomerulus", 255: "Glomerulus"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _get_patient_id(self, img_file):
        patient_id = os.path.basename(img_file).replace(".tif", "").replace("image", "")
        return patient_id

class MouseKidneySRX_revised(BasicDataset):
    def __init__(self, data_dir="datasets/SRX/MouseKidneySRX_revised", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.mask_files = glob(os.path.join(self.data_dir, "mask", "*.tiff"))
        self.img_files = [mask_file.replace("mask", "image") for mask_file in self.mask_files]
        self.labels_dict = {1: "Glomerulus"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _get_patient_id(self, img_file):
        patient_id = os.path.basename(img_file).replace(".tiff", "").replace("image", "")
        return patient_id

class SNEMI3D(BasicDataset):
    def __init__(self, data_dir="datasets/EM/SNEMI3D", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.mask_files = glob(os.path.join(self.data_dir, "seg*.tif"))
        self.img_files = [mask_file.replace("seg", "image") for mask_file in self.mask_files]
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _read_image_mask(self, index):
        img_file = self.img_files[index]
        mask_file = self.mask_files[index]
        img, img_array, spacing = self._read_nii_file(img_file)
        mask, mask_array, _ = self._read_nii_file(mask_file)
        self.labels_dict = {i: "NeuronInstance" for i in np.unique(mask_array) if i != 0}
        if img_array.shape != mask_array.shape:
            self.printer(f"!!! Failed to load img={img_file}\tmask={mask_file} [img={img_array.shape}\tmask={mask_array.shape}]")
            return None, None, None, None, None, None, None
        return img_file, mask_file, img, img_array, mask, mask_array, spacing

    def _get_patient_id(self, img_file):
        patient_id = os.path.basename(img_file).replace(".tif", "").replace("image", "")
        return patient_id

class GlomSegmicroCT(BasicDataset):
    def __init__(self, data_dir="datasets/microCT/GlomSeg-microCT", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.mask_files = glob(os.path.join(self.data_dir, "*.nrrd"))
        self.img_files = [mask_file.replace("-segmentation.seg.nrrd", ".tiff") for mask_file in self.mask_files]
        self.labels_dict = {1: "Glomerulus"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _get_patient_id(self, img_file):
        patient_id = os.path.basename(img_file).replace(".tiff", "")
        return patient_id

class TongueCancerBP(BasicDataset):
    def __init__(self, data_dir="datasets/CT/TongueCancerBP", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.mask_files = glob(os.path.join(self.data_dir, "*", "Segmentation.seg.nrrd"))
        self.img_files = []
        for mask_file in self.mask_files:
            mask_array = sitk.GetArrayFromImage(sitk.ReadImage(mask_file))
            img_files = glob(mask_file.replace("Segmentation.seg.nrrd", "*.nii"))
            flag = False
            for img_file in img_files:
                img_array = sitk.GetArrayFromImage(sitk.ReadImage(img_file))
                if img_array.shape[0] == mask_array.shape[0]:
                    self.img_files.append(img_file)
                    flag = True 
                    break
            if flag == False:
                print(mask_file, "!!!!!!!")
        self.labels_dict = {1: "tonguecancer"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _get_patient_id(self, img_file):
        patient_id = img_file.split("/")[-2]
        return patient_id

class TongueCancer0807(BasicDataset):
    def __init__(self, data_dir="datasets/CT/TongueCancer0807", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.mask_files = glob(os.path.join(self.data_dir, "masks", "*_mask.nii.gz"))
        self.img_files = [mask_file.replace("masks", "images").replace("_mask.nii.gz", "_image.nii.gz") for mask_file in self.mask_files]
        self.labels_dict = {1: "tonguecancer"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _get_patient_id(self, img_file):
        patient_id = os.path.basename(img_file).split("_")[0]
        return patient_id

# finetune dataset (inference val data only)
class FT_GlomSegmicroCT(GlomSegmicroCT):
    def __init__(self, data_dir="datasets/microCT/GlomSeg-microCT", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        val_data = pd.read_csv("data/finetune_data_split/GlomSeg_microCT/Val.csv").values
        self.keep_patient_ids = list(set([os.path.basename(task_dir).split("_")[0] for _, _, _, task_dir in val_data]))
        mask_files = glob(os.path.join(self.data_dir, "*.nrrd"))
        img_files = [mask_file.replace("-segmentation.seg.nrrd", ".tiff") for mask_file in self.mask_files]
        self.img_files, self.mask_files = [], []
        for img_file, mask_file in zip(img_files, mask_files):
            patient_id = self._get_patient_id(img_file)
            if patient_id in self.keep_patient_ids:
                self.img_files.append(img_file)
                self.mask_files.append(mask_file)
        self.labels_dict = {1: "Glomerulus"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

class FT_AdrenalACCKi67Seg(AdrenalACCKi67Seg):
    def __init__(self, data_dir="datasets/CT/Adrenal-ACC-Ki67-Seg/manifest-1680809675630/Adrenal-ACC-Ki67-Seg/", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        segmentation_folders = []
        for root, dirs, files in os.walk(self.data_dir):
            for dir in dirs:
                if "Segmentation" in dir:
                    full_path = os.path.join(root, dir)
                    segmentation_folders.append(full_path)
        self.segmentation_folders = []
        val_data = pd.read_csv("data/finetune_data_split/Adrenal-ACC-Ki67-Seg/Val.csv").values
        self.keep_patient_ids = list(set([os.path.basename(task_dir).split("_")[0] for _, _, _, task_dir in val_data]))
        for segmentation_folder in segmentation_folders:
            patient_id = self._get_patient_id(segmentation_folder)
            if patient_id in self.keep_patient_ids:
                self.segmentation_folders.append(segmentation_folder)
        self.labels_dict = {255: "adrenocorticalcarcinoma"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

class FT_CHAOSCT(CHAOSCT):
    def __init__(self, data_dir="datasets/CT/CHAOS-CT", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        raw_data_dirs = glob(os.path.join(self.data_dir, "Train_Sets/CT/*/DICOM_anon/"))
        self.raw_data_dirs = []
        val_data = pd.read_csv("data/finetune_data_split/CHAOS-CT/Val.csv").values
        self.keep_patient_ids = list(set([os.path.basename(task_dir).split("_")[0] for _, _, _, task_dir in val_data]))
        for raw_data_dir in raw_data_dirs:
            patient_id = self._get_patient_id(raw_data_dir)
            if patient_id in self.keep_patient_ids:
                self.raw_data_dirs.append(raw_data_dir)
        self.labels_dict = {1: "liver"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

class FT_HCCTACESeg(HCCTACESeg):
    def __init__(self, data_dir="datasets/CT/HCC-TACE-Seg", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        val_data = pd.read_csv("data/finetune_data_split/HCC-TACE-Seg/Val.csv").values
        self.keep_patient_ids = list(set([os.path.basename(task_dir).split("_")[0] for _, _, _, task_dir in val_data]))
        img_files = glob(os.path.join(self.data_dir, "preprocessed/IMG/*_IMG.nii.gz"))
        mask_files = [img_file.replace("IMG", "SEG") for img_file in img_files]
        self.img_files, self.mask_files = [], []
        for img_file, mask_file in zip(img_files, mask_files):
            patient_id = self._get_patient_id(img_file)
            if patient_id in self.keep_patient_ids:
                self.img_files.append(img_file)
                self.mask_files.append(mask_file)
        self.labels_dict = {1: "liver", 2: "livertumor", 3: "liverbloodvessel"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

class FT_LNQ2023(LNQ2023):
    def __init__(self, data_dir="datasets/CT/LNQ2023", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        val_data = pd.read_csv("data/finetune_data_split/LNQ2023/Val.csv").values
        self.keep_patient_ids = list(set([os.path.basename(task_dir).split("_")[0] for _, _, _, task_dir in val_data]))
        mask_files = glob(os.path.join(self.data_dir, "labelsTr/*.nii.gz"))
        img_files = [mask_file.replace("labelsTr", "imagesTr").replace(".nii.gz", "_0000.nii.gz") for mask_file in mask_files]
        self.img_files, self.mask_files = [], []
        for img_file, mask_file in zip(img_files, mask_files):
            patient_id = self._get_patient_id(img_file)
            if patient_id in self.keep_patient_ids:
                self.img_files.append(img_file)
                self.mask_files.append(mask_file)
        self.labels_dict = {1: "lnq"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _get_patient_id(self, img_file):
        patient_id = os.path.basename(img_file).split('.')[0].split("_")[0]
        return patient_id

class FT_WORD(WORD):
    def __init__(self, data_dir="datasets/CT/WORD", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir
        val_data = pd.read_csv("data/finetune_data_split/WORD/Val.csv").values
        self.keep_patient_ids = list(set([os.path.basename(task_dir).split("_")[0] for _, _, _, task_dir in val_data]))
        img_files = glob(os.path.join(self.data_dir, "WORD-V0.1.0/imagesTr/*.nii.gz")) \
                    + glob(os.path.join(self.data_dir, "WORD-V0.1.0/imagesVal/*.nii.gz"))
        mask_files =  [img_file.replace("images", "labels") for img_file in img_files]
        self.img_files, self.mask_files = [], []
        for img_file, mask_file in zip(img_files, mask_files):
            patient_id = self._get_patient_id(img_file)
            if patient_id in self.keep_patient_ids:
                self.img_files.append(img_file)
                self.mask_files.append(mask_file)
        self.labels_dict = {1: "liver", 2: "spleen", 3: "left_kidney", 4: "right_kidney",
                            5: "stomach", 6: "gallbladder", 7: "esophagus", 8: "pancreas",
                            9: "duodenum", 10: "colon", 11: "intestine", 12: "adrenal",
                            13: "rectum", 14: "bladder", 15: "Head_of_femur_L", 16: "Head_of_femur_R"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

class FT_HaNSegCT(HaNSegCT):
    def __init__(self, data_dir="datasets/CT/HaN-Seg-CT", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        val_data = pd.read_csv("data/finetune_data_split/HaN-Seg-CT/Val.csv").values
        self.keep_patient_ids = list(set([os.path.basename(task_dir).split("_")[0] for _, _, _, task_dir in val_data]))
        mask_files = glob(os.path.join(self.data_dir, "preprocessed/SEG/*/*.seg.nrrd"))
        img_files = [mask_file.replace("SEG", "IMG") for mask_file in self.mask_files]
        img_files = [img_file.replace(os.path.basename(img_file), "_".join([*os.path.basename(img_file).split("_")[:2], "IMG", "CT"])+".nrrd") for img_file in img_files]
        self.img_files, self.mask_files = [], []
        for img_file, mask_file in zip(img_files, mask_files):
            patient_id = self._get_patient_id(img_file)
            if patient_id in self.keep_patient_ids:
                self.img_files.append(img_file)
                self.mask_files.append(mask_file)
        self.printer(f"[{data_dir}] {self.__len__()} samples")

class FT_ACDC(ACDC):
    def __init__(self, data_dir="datasets/MR/ACDC", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        val_data = pd.read_csv("data/finetune_data_split/ACDC/Val.csv").values
        self.keep_patient_ids = list(set([os.path.basename(task_dir).split("_")[0] for _, _, _, task_dir in val_data]))
        mask_files = glob(os.path.join(self.data_dir, "training/*/*_gt.nii.gz")) \
                        + glob(os.path.join(self.data_dir, "testing/*/*_gt.nii.gz"))
        img_files =  [mask_file.replace("_gt.nii.gz", ".nii.gz") for mask_file in mask_files]
        self.img_files, self.mask_files = [], []
        for img_file, mask_file in zip(img_files, mask_files):
            patient_id = self._get_patient_id(img_file)
            if patient_id in self.keep_patient_ids:
                self.img_files.append(img_file)
                self.mask_files.append(mask_file)
        self.labels_dict = {1: "rightventricle", 2: "myocardium", 3: "leftventricle"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

class FT_CHAOSMR(CHAOSMR):
    def __init__(self, data_dir="datasets/MR/CHAOS-MR", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        val_data = pd.read_csv("data/finetune_data_split/CHAOS-MR/Val.csv").values
        self.keep_patient_ids = list(set([os.path.basename(task_dir).split("_")[0] for _, _, _, task_dir in val_data]))
        img_dirs = glob(os.path.join(self.data_dir, "Train_Sets/MR/*/T1DUAL/DICOM_anon/InPhase/")) \
            + glob(os.path.join(self.data_dir, "/Train_Sets/MR/*/T2SPIR/DICOM_anon/"))
        mask_dirs = []
        for img_dir in img_dirs:
            if "T1DUAL" in img_dir:
                mask_dir = img_dir.replace("InPhase/", "").replace("OutPhase/", "").replace("DICOM_anon", "Ground")
            else:
                mask_dir = img_dir.replace("DICOM_anon", "Ground")
            mask_dirs.append(mask_dir)
        self.img_dirs, self.mask_dirs = [], []
        for img_dir, mask_dir in zip(img_dirs, mask_dirs):
            patient_id = self._get_patient_id(img_dir)
            if patient_id in self.keep_patient_ids:
                self.img_dirs.append(img_dir)
                self.mask_dirs.append(mask_dir)
        self.labels_dict = {63: "liver", 126: "leftkidney", 189: "rightkidney", 252: "spleen"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

class FT_QUBIQCT(QUBIQCT):
    def __init__(self, data_dir="datasets/CT/QUBIQ-CT", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        val_data = pd.read_csv("data/finetune_data_split/QUBIQ-CT/Val.csv").values
        self.keep_patient_ids = list(set([os.path.basename(task_dir).split("_")[0] for _, _, _, task_dir in val_data]))
        mask_files = glob(os.path.join(self.data_dir, "Training/*/*/*seg*.nii.gz")) \
                        + glob(os.path.join(self.data_dir, "Validation/*/Validation/*/*seg*.nii.gz"))
        img_files =[mask_file.replace(os.path.basename(mask_file), "image.nii.gz") for mask_file in mask_files]
        self.img_files, self.mask_files = [], []
        for img_file, mask_file in zip(img_files, mask_files):
            patient_id = self._get_patient_id(img_file)
            if patient_id in self.keep_patient_ids:
                self.img_files.append(img_file)
                self.mask_files.append(mask_file)
        self.printer(f"[{data_dir}] {self.__len__()} samples")

class FT_TongueCancer0807(TongueCancer0807):
    def __init__(self, data_dir="datasets/CT/TongueCancer0807", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        val_data = pd.read_csv("data/finetune_data_split/TongueCancer0807/Val.csv").values
        self.keep_patient_ids = list(set([os.path.basename(task_dir).split("_")[0] for _, _, _, task_dir in val_data]))
        mask_files = glob(os.path.join(self.data_dir, "masks", "*_mask.nii.gz"))
        img_files = [mask_file.replace("masks", "images").replace("_mask.nii.gz", "_image.nii.gz") for mask_file in self.mask_files]
        self.img_files, self.mask_files = [], []
        for img_file, mask_file in zip(img_files, mask_files):
            patient_id = self._get_patient_id(img_file)
            if patient_id in self.keep_patient_ids:
                self.img_files.append(img_file)
                self.mask_files.append(mask_file)
        self.labels_dict = {1: "tonguecancer"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _get_patient_id(self, img_file):
        patient_id = os.path.basename(img_file).split("_")[0]
        return patient_id

class FT_MouseKidneySRX(MouseKidneySRX):
    def __init__(self, data_dir="datasets/SRX/MouseKidneySRX", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        val_data = pd.read_csv("data/finetune_data_split/MouseKidneySRX/Val.csv").values
        self.keep_patient_ids = list(set([os.path.basename(task_dir).split("_")[0] for _, _, _, task_dir in val_data]))
        mask_files = glob(os.path.join(self.data_dir, "seg*.tif"))
        img_files = [mask_file.replace("seg", "image") for mask_file in self.mask_files]
        self.img_files, self.mask_files = [], []
        for img_file, mask_file in zip(img_files, mask_files):
            patient_id = self._get_patient_id(img_file)
            if patient_id in self.keep_patient_ids:
                self.img_files.append(img_file)
                self.mask_files.append(mask_file)
        self.labels_dict = {1: "Glomerulus", 255: "Glomerulus"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

class FT_MouseKidneySRX_revised(MouseKidneySRX_revised):
    def __init__(self, data_dir="datasets/SRX/MouseKidneySRX_revised", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        val_data = pd.read_csv("data/finetune_data_split/MouseKidneySRX/Val.csv").values
        self.keep_patient_ids = list(set([os.path.basename(task_dir).split("_")[0] for _, _, _, task_dir in val_data]))
        mask_files = glob(os.path.join(self.data_dir, "mask", "*.tiff"))
        img_files = [mask_file.replace("mask", "image") for mask_file in self.mask_files]
        self.img_files, self.mask_files = [], []
        for img_file, mask_file in zip(img_files, mask_files):
            patient_id = self._get_patient_id(img_file)
            if patient_id in self.keep_patient_ids:
                self.img_files.append(img_file)
                self.mask_files.append(mask_file)
        self.labels_dict = {1: "Glomerulus"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

class LYTGEJ0809(BasicDataset):
    def __init__(self, data_dir="datasets/CT/LYT_GEJ_0809/Datasets_0809/Datasets_0809", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.mask_files = glob(os.path.join(self.data_dir, "masks/*.nii.gz"))
        self.img_files = [mask_file.replace("masks", "images") for mask_file in self.mask_files]
        self.labels_dict = {1: "GEJ"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _get_patient_id(self, img_file):
        patient_id = os.path.basename(img_file).split(".")[0]
        return patient_id

class LYTGEJ0809E(BasicDataset):
    def __init__(self, data_dir="datasets/CT/LYT_GEJ_0809/Datasets_0809/Datasets_0809_e", instance_threshold=200, printer=print):
        super().__init__(data_dir=data_dir, instance_threshold=instance_threshold, printer=printer)
        self.data_dir = data_dir 
        self.mask_files = glob(os.path.join(self.data_dir, "masks/*.nii.gz"))
        self.img_files = [mask_file.replace("masks", "images").replace("Datasets_0809_e", "Datasets_0809") for mask_file in self.mask_files]
        self.labels_dict = {1: "LYTE", 2: "LYTE", 3: "LYTE"}
        self.printer(f"[{data_dir}] {self.__len__()} samples")

    def _get_patient_id(self, img_file):
        patient_id = os.path.basename(img_file).split(".")[0]
        return patient_id

def build_dataset(args, printer=print):
    if args.dataset == "AbdomenCT-1K":
        ds = AbdomenCT1k(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "Adrenal-ACC-Ki67-Seg":
        ds = AdrenalACCKi67Seg(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "AMOS-CT":
        ds = AMOSCT(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "AutoPET-CT":
        ds = AutoPETCT(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "AutoPET-PETCT":
        ds = AutoPETPETCT(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "COVID-19 Seg. Challenge":
        ds = COVID19SegChallenge(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "COVID-19-CT-Seg":
        ds = COVID19CTSeg(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "CHAOS-CT":
        ds = CHAOSCT(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "HCC-TACE-Seg":
        ds = HCCTACESeg(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "HECKTOR":
        ds = HECKTOR(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "INSTANCE":
        ds = INSTANCE(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "KiPA":
        ds = KiPA(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "KiTS":
        ds = KiTS(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "Lymph Nodes":
        ds = LymphNodes(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "LNQ2023":
        ds = LNQ2023(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "Task03_Liver":
        ds = Task03Liver(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "Task06_Lung":
        ds = Task06Lung(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "Task07_Pancreas":
        ds = Task07Pancreas(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "Task08_HepaticVessel":
        ds = Task08HepaticVessel(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "Task09_Spleen":
        ds = Task09Spleen(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "Task10_Colon":
        ds = Task10Colon(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "NSCLC Pleural Effusion":
        ds = NSCLCPleuralEffusion(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "TotalSegmentator":
        ds = TotalSegmentator(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "WORD":
        ds = WORD(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "HaN-Seg-CT":
        ds = HaNSegCT(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "QUBIQ-CT":
        ds = QUBIQCT(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "ACDC":
        ds = ACDC(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "AMOS-MR":
        ds = AMOSMR(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "ATLAS-R2.0":
        ds = ATLASR20(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "BraTS":
        ds = BraTS(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "CC-Tumor-Heterogeneity":
        ds = CCTumorHeterogeneity(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "CHAOS-MR":
        ds = CHAOSMR(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "ISLES":
        ds = ISLES(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "MnM2":
        ds = MnM2(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "Task01_BrainTumour":
        ds = Task01BrainTumour(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "Task02_Heart":
        ds = Task02Heart(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "Task04_Hippocampus":
        ds = Task04Hippocampus(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "Task05_Prostate":
        ds = Task05Prostate(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "Spine":
        ds = Spine(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "Qin-Prostate-Repeatability":
        ds = QinProstateRepeatability(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "PROMISE":
        ds = PROMISE(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "PI-CAI":
        ds = PICAI(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "WMH":
        ds = WMH(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "NCI-ISBI":
        ds = NCIISBI(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "HM-3d-Seg":
        ds = HM3dSeg(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "TCGA-STAD":
        ds = TCGASTAD(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "GCSeg_PKCancer":
        ds = GCSegPKCancer(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "GlomSeg_microCT":
        ds = GlomSegmicroCT(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "TongueCancerBP":
        ds = TongueCancerBP(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "FT_GlomSeg_microCT":
        ds = FT_GlomSegmicroCT(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "FT_Adrenal-ACC-Ki67-Seg":
        ds = FT_AdrenalACCKi67Seg(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "FT_CHAOS-CT":
        ds = FT_CHAOSCT(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "FT_HCC-TACE-Seg":
        ds = FT_HCCTACESeg(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "FT_LNQ2023":
        ds = FT_LNQ2023(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "FT_WORD":
        ds = FT_WORD(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "FT_HaN-Seg-CT":
        ds = FT_HaNSegCT(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "FT_ACDC":
        ds = FT_ACDC(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "FT_CHAOS-MR":
        ds = FT_CHAOSMR(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "FT_QUBIQ-CT":
        ds = FT_QUBIQCT(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "TongueCancer0807":
        ds = TongueCancer0807(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "LYT_GEJ_0809":
        ds = LYTGEJ0809(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "LYT_GEJ_0809_e":
        ds = LYTGEJ0809E(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "FT_TongueCancer0807":
        ds = FT_TongueCancer0807(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "MouseKidneySRX":
        ds = MouseKidneySRX(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "FT_MouseKidneySRX":
        ds = FT_MouseKidneySRX(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "SNEMI3D":
        ds = SNEMI3D(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "MouseKidneySRX_revised":
        ds = MouseKidneySRX_revised(instance_threshold=args.instance_threshold, printer=printer)
    elif args.dataset == "FT_MouseKidneySRX_revised":
        ds = FT_MouseKidneySRX_revised(instance_threshold=args.instance_threshold, printer=printer)
    else:
        raise NotImplementedError(ds)
    return ds

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="COVID-19-CT-Seg")
    parser.add_argument('--model', type=str, default="PropNetV1")
    parser.add_argument('--checkpoint', type=str, default="checkpoints/propsam_1200+600+400+1200+1100_512c.pth")
    parser.add_argument('--box2seg_model', type=str, default="Box2SegNet")
    parser.add_argument('--box2seg_checkpoint', type=str, default="checkpoints/box2segplus_1400+1400+1400.pth")
    parser.add_argument('--box2seg_percentile_low_start', type=float, default=5)
    parser.add_argument('--box2seg_percentile_low_end', type=float, default=21)
    parser.add_argument('--box2seg_percentile_low_interval', type=float, default=1)
    parser.add_argument('--box2seg_percentile_high_start', type=float, default=90)
    parser.add_argument('--box2seg_percentile_high_end', type=float, default=96)
    parser.add_argument('--box2seg_percentile_high_interval', type=float, default=1)
    parser.add_argument('--input_channels', type=int, default=1)
    parser.add_argument('--n_stages', type=int, default=6)
    parser.add_argument('--deep_supervision', type=bool, default=True)
    parser.add_argument('--n_attn_stage', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--conv_dim', type=int, default=2)
    parser.add_argument('--max_channels', type=int, default=512)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--target_size', type=int, default=224)
    parser.add_argument('--dynamic_crop_ratio', type=float, default=1.25)
    parser.add_argument('--max_Z', type=int, default=10)
    parser.add_argument('--neighbor_spacing_z', type=int, default=20)
    parser.add_argument('--instance_threshold', type=int, default=100)
    parser.add_argument('--eval_dir', type=str, default="data/step3_eval_data/valid")
    parser.add_argument('--metric_save_file', type=str, default="valid.csv")
    parser.add_argument('--from_scratch_ratio', type=float, default=0.0)
    parser.add_argument('--infer_dir', type=str, default="infer_with_box2seg")
    parser.add_argument('--start_i', type=int, default=0)
    parser.add_argument('--end_i', type=int, default=-1)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = build_args()
    set_seed(args.seed)
    # printer = print

    dataset_infer_dir = os.path.join(args.infer_dir, args.dataset)
    os.makedirs(dataset_infer_dir, exist_ok=True)
    logger = build_logger(os.path.join(dataset_infer_dir, "log.log"))
    printer = logger.info
    log_parser_args(args, printer)
    dataset_infer_dir = os.path.join(dataset_infer_dir, "data")
    os.makedirs(dataset_infer_dir, exist_ok=True)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    propnet = eval(args.model)(args) 
    printer(f"Total trainable parameters: {1.0*count_parameters(propnet)/1e6:.2f}M")
    propnet = load_checkpoint(propnet, args.checkpoint, printer=print)
    propnet = propnet.to(args.device)

    if args.box2seg_model in ["Box2SegNet",]:
        box2seg = eval(args.box2seg_model)(args)
        printer(f"[Seg2box] Total trainable parameters: {1.0*count_parameters(box2seg)/1e6:.2f}M")
        box2seg = load_checkpoint(box2seg, args.box2seg_checkpoint, printer=print)
        box2seg = box2seg.to(args.device)
    else:
        box2seg = None

    records = {"dice": []}
    ds = build_dataset(args, printer=printer)
    num_ignored = 0
    for i in range(len(ds)):
        if i < args.start_i: continue
        if args.end_i > 0 and i >= args.end_i: break
        # if i >= 20: break
        # if i < 98: continue
        patient_id, img_file, img, img_array, spacing, mask, mask_array, support_category_ids, support_categories, support_mask_arrays, support_zs = ds.get_image_mask(i)
        if img is None:
            printer(f"img is None")
            printer("=" * 100)
            num_ignored += 1
            continue
        if len(support_mask_arrays) == 0: 
            printer(f"!!! Ignored, there are not targets ({len(support_mask_arrays)}), {img_file}")
            printer("=" * 100)
            num_ignored += 1
            continue
        used_time, pred_array = infer(args, ds.get_modality(), box2seg, propnet, img, img_array, spacing, 
                mask, support_category_ids, support_categories, support_mask_arrays, support_zs, 
                None, args.target_size, args.dynamic_crop_ratio, args.neighbor_spacing_z, args.max_Z, args.device, printer)
        sample_metric_dict = calculate_metrics(pred_array, mask_array, ds.labels_dict)
        sample_metric_dict["used_time"] = used_time
        for key in sample_metric_dict:
            if key not in records:
                records[key] = []
            records[key].append(sample_metric_dict[key])
            if "dice" in key:
                records["dice"].append(sample_metric_dict[key])
        printer(f"\tsample={sample_metric_dict}")
        print_context = "\t".join(f"{key}={np.mean(value):.4f}" for key, value in records.items())
        printer(f"{i}\t{print_context}")

        save_name = img_file.replace("/", "___").split(".")[0] + "___" + patient_id + "___" + str(int(i))
        printer(f"Save to {save_name}")
        img_file = ds._read_image_mask(i)[0]
        img_nii = sitk.GetImageFromArray(img_array)
        img_nii.SetSpacing(spacing)
        sitk.WriteImage(img_nii, os.path.join(dataset_infer_dir, save_name+".img.nii.gz"))
        pred_nii = sitk.GetImageFromArray(pred_array)
        pred_nii.SetSpacing(spacing)
        sitk.WriteImage(pred_nii, os.path.join(dataset_infer_dir, save_name+".pred.nii.gz"))
        mask_nii = sitk.GetImageFromArray(mask_array)
        mask_nii.SetSpacing(spacing)
        sitk.WriteImage(mask_nii, os.path.join(dataset_infer_dir, save_name+".mask.nii.gz"))
        printer("=" * 100)

    printer(f"A total of {ds.__len__()} samples !!! Ignored {num_ignored} samples !!!")



