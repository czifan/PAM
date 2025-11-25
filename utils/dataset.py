import os
import cv2 
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from glob import glob 
import torch 
import matplotlib.pyplot as plt 
import pandas as pd
from scipy.ndimage import label
from PIL import Image
import random

class PropDataset(Dataset):
    def __init__(self, args, split_file, split, printer=print):
        super().__init__()
        self.data_dir = args.data_dir 
        self.split = split 
        self.sampling_N_queries = args.sampling_N_queries
        self.center_crop_size = args.center_crop_size
        self.target_size = args.target_size
        self.train_iter_samples = args.train_iter_samples
        self.eval_iter_samples = args.eval_iter_samples
        self.samples = pd.read_csv(split_file).values
        self.len_samples = len(self.samples)
        printer(f"[{self.split}] {split_file}: {self.len_samples} samples")

        if self.split in ["train",]:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Resize(width=self.target_size, height=self.target_size, p=1.0),
                #A.RandomCrop(width=self.target_size, height=self.target_size, p=1.0), 
                A.Rotate(limit=45, p=0.5),
                A.Normalize(mean=(0.5), std=(0.5)), 
                ToTensorV2(), 
            ])
            self.min_crop_scale = 1.0
            self.max_crop_scale = 2.0
        else:
            self.transform = A.Compose([
                #A.CenterCrop(width=self.target_size, height=self.target_size, p=1.0),
                A.Resize(width=self.target_size, height=self.target_size, p=1.0),
                A.Normalize(mean=[0.5], std=(0.5)), 
                ToTensorV2(),
            ])
            self.min_crop_scale = 1.5
            self.max_crop_scale = 1.5

        printer(f"[{self.split}] Loaded {self.__len__()} samples")
    
    def __len__(self):
        if self.split in ["train",] and self.train_iter_samples > 0:
            return self.train_iter_samples
        elif self.split not in ["train",] and self.eval_iter_samples > 0:
            return self.eval_iter_samples
        else:
            return len(self.samples)

    def _crop_with_resize(self, image, x1, y1, x2, y2, crop_size, is_mask):
        cropped_image = image[y1:y2, x1:x2]
        if cropped_image.shape[0] < crop_size[0] or cropped_image.shape[1] < crop_size[1]:
            cropped_image = cv2.copyMakeBorder(cropped_image, top=0, bottom=crop_size[0] - cropped_image.shape[0],
                                               left=0, right=crop_size[1] - cropped_image.shape[1],
                                               borderType=cv2.BORDER_CONSTANT, value=0)
        return cropped_image

    def _center_crop(self, image, mask, query_images, query_masks, min_crop_scale, max_crop_scale, min_crop_size=64):
        H, W = mask.shape 
        inds = np.where(mask)
        y1, x1, y2, x2 = np.min(inds[0]), np.min(inds[1]), np.max(inds[0]), np.max(inds[1])
        center_y = int((y1 + y2) / 2.0)
        center_x = int((x1 + x2) / 2.0)
        h = y2 - y1
        w = x2 - x1
        crop_size = int(np.random.uniform(min_crop_scale, max_crop_scale) * max(h, w))
        crop_size = max(crop_size, min_crop_size)
        crop_size = (crop_size, crop_size)

        x1 = max(center_x - crop_size[1] // 2, 0)
        y1 = max(center_y - crop_size[0] // 2, 0)
        x2 = min(x1 + crop_size[1], W)
        y2 = min(y1 + crop_size[0], H)

        cropped_image = self._crop_with_resize(image, x1, y1, x2, y2, crop_size, False)
        cropped_mask = self._crop_with_resize(mask, x1, y1, x2, y2, crop_size, True)
        cropped_query_images = [self._crop_with_resize(query_image, x1, y1, x2, y2, crop_size, False) for query_image in query_images]
        cropped_query_masks = [self._crop_with_resize(query_mask, x1, y1, x2, y2, crop_size, True) for query_mask in query_masks]

        return cropped_image, cropped_mask, cropped_query_images, cropped_query_masks
    
    def __getitem__(self, idx):
        if self.split in ["train",] and self.train_iter_samples > 0:
            random_idx = np.random.randint(0, self.len_samples)
            sample = self.samples[random_idx]
        elif self.split not in ["train",] and self.eval_iter_samples > 0:
            random_idx = np.random.randint(0, self.len_samples)
            sample = self.samples[random_idx]
        else:
            sample = self.samples[idx]
        dataset_name, _, object_name, sample_id = sample
        modality, sample_id = sample_id.split("/", 1)
        sample_dir = os.path.join(self.data_dir, modality, sample_id)
        # print(sample_dir, "+++++", len(os.listdir(sample_dir)))
        if self.data_dir == "nephron_segmentation/nephron_lhf/task_data_nsz20":
            support_x_path = os.path.join(sample_dir, f"support_-1_img.jpg")
            support_y_path = os.path.join(sample_dir, f"support_-1_mask.jpg")
        else:
            slice_id = int(sample_id.split("/")[1].split("_")[1])
            support_x_path = os.path.join(sample_dir, f"support_{slice_id:03d}_img.jpg")
            support_y_path = os.path.join(sample_dir, f"support_{slice_id:03d}_mask.jpg")
        query_x_paths = np.random.choice(glob(os.path.join(sample_dir, f"query_*_img.jpg")), self.sampling_N_queries, replace=True)
        query_y_paths = [query_x_path.replace("_img", "_mask") for query_x_path in query_x_paths]

        support_x = cv2.imread(support_x_path, cv2.IMREAD_GRAYSCALE)
        support_y = cv2.imread(support_y_path, cv2.IMREAD_GRAYSCALE)
        query_xs = [cv2.imread(query_x_path, cv2.IMREAD_GRAYSCALE) for query_x_path in query_x_paths]
        query_ys = [cv2.imread(query_y_path, cv2.IMREAD_GRAYSCALE) for query_y_path in query_y_paths]
        support_x, support_y, query_xs, query_ys = self._center_crop(
            support_x, support_y, query_xs, query_ys, min_crop_scale=self.min_crop_scale, max_crop_scale=self.max_crop_scale)
        
        support_augmented = self.transform(image=support_x, mask=support_y)
        support_x = support_augmented['image'] # (1, 224, 224)
        support_y = support_augmented['mask'].float() / 255.0 # (224, 224)
        if torch.max(support_y).item() == 0:
            return self.__getitem__((idx+1)%self.__len__())
        for i, (query_x, query_y) in enumerate(zip(query_xs, query_ys)):
            query_augmented = self.transform(image=query_x, mask=query_y)
            query_x = query_augmented['image']
            query_y = query_augmented['mask']
            query_xs[i] = query_x
            query_ys[i] = query_y
        query_xs = torch.stack(query_xs, dim=0) # (N_queries, 1, 224, 224)
        query_ys = torch.stack(query_ys, dim=0).float() / 255.0 # (N_queries, 224, 224)

        return {
            "modality": modality,
            "sample_id": sample_id,
            "support_x": support_x,
            "support_y": support_y,
            "query_xs": query_xs,
            "query_ys": query_ys,
            "dataset_name": dataset_name,
            "object_name": object_name,
        }

def show_task(support_x, support_y, query_xs, query_ys, save_file):
    support_x = support_x.detach().cpu().numpy()
    support_y = support_y.detach().cpu().numpy()
    query_xs = query_xs.detach().cpu().numpy()
    query_ys = query_ys.detach().cpu().numpy()
    N = 1 + query_xs.shape[0]
    _, ax = plt.subplots(N, 2, figsize=(10, 5*N))
    ax[0][0].imshow(support_x[0]*0.5+0.5, cmap="gray")
    ax[0][1].imshow(support_y, cmap="gray")
    for i in range(query_xs.shape[0]):
        ax[i+1][0].imshow(query_xs[i][0]*0.5+0.5, cmap="gray")
        ax[i+1][1].imshow(query_ys[i], cmap="gray")
    plt.savefig(save_file)
    plt.close()

class Box2SegDataset(Dataset):
    def __init__(self, args, split_file, split, printer=print):
        super().__init__()
        self.data_dir = args.data_dir 
        self.split = split 
        self.sampling_N_queries = args.sampling_N_queries
        self.center_crop_size = args.center_crop_size
        self.target_size = args.target_size
        self.train_iter_samples = args.train_iter_samples
        self.eval_iter_samples = args.eval_iter_samples
        self.img_files = open(split_file, "r").readlines()
        self.len_samples = len(self.img_files)

        if self.split in ["train",]:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Resize(width=self.target_size, height=self.target_size, p=1.0),
                #A.RandomCrop(width=self.target_size, height=self.target_size, p=1.0), 
                A.RandomBrightnessContrast(contrast_limit=0.2, p=0.5),
                A.Rotate(limit=45, p=0.5),
                A.Normalize(mean=(0.5), std=(0.5)), 
                ToTensorV2(), 
            ])
            self.min_crop_scale = 1.0
            self.max_crop_scale = 1.25
        else:
            self.transform = A.Compose([
                #A.CenterCrop(width=self.target_size, height=self.target_size, p=1.0),
                A.Resize(width=self.target_size, height=self.target_size, p=1.0),
                A.Normalize(mean=[0.5], std=(0.5)), 
                ToTensorV2(),
            ])
            self.min_crop_scale = 1.0
            self.max_crop_scale = 1.01

        printer(f"[{self.split}] {split_file}: {self.len_samples} samples | sampling {self.__len__()} samples")

    def __len__(self):
        if self.split in ["train",] and self.train_iter_samples > 0:
            return self.train_iter_samples
        elif self.split not in ["train",] and self.eval_iter_samples > 0:
            return self.eval_iter_samples
        else:
            return len(self.samples)

    def _crop_with_resize(self, image, x1, y1, x2, y2, crop_size, is_mask):
        cropped_image = image[y1:y2, x1:x2]
        cropped_image = cv2.resize(cropped_image, crop_size)
        return cropped_image

    def _center_crop(self, image, mask, min_crop_scale, max_crop_scale, min_crop_size=16):
        structure = np.ones((3, 3), dtype=mask.dtype) 
        labeled_mask, num_features = label(mask, structure=structure)

        if num_features == 0:
            raise ValueError("No connected components found in the mask.")

        feature_index = random.randint(1, num_features)
        selected_mask = (labeled_mask == feature_index)

        inds = np.where(selected_mask)
        y1, x1, y2, x2 = np.min(inds[0]), np.min(inds[1]), np.max(inds[0]), np.max(inds[1])
        center_y = int((y1 + y2) / 2.0)
        center_x = int((x1 + x2) / 2.0)
        h = y2 - y1
        w = x2 - x1

        crop_size = int(np.random.uniform(min_crop_scale, max_crop_scale) * max(h, w))
        crop_size = max(crop_size, min_crop_size)
        crop_size = (crop_size, crop_size)

        x1 = max(center_x - crop_size[1] // 2, 0)
        y1 = max(center_y - crop_size[0] // 2, 0)
        x2 = min(x1 + crop_size[1], mask.shape[1])
        y2 = min(y1 + crop_size[0], mask.shape[0])

        cropped_image = self._crop_with_resize(image, x1, y1, x2, y2, crop_size, False)
        cropped_mask = self._crop_with_resize(mask, x1, y1, x2, y2, crop_size, True)

        return cropped_image, cropped_mask

    def __getitem__(self, idx):
        #img_file = self.img_files[idx].split(",")[0]
        if self.split in ["train",] and self.train_iter_samples > 0:
            random_idx = np.random.randint(0, self.len_samples)
            img_file = self.img_files[random_idx]
        elif self.split not in ["train",] and self.eval_iter_samples > 0:
            random_idx = np.random.randint(0, self.len_samples)
            img_file = self.img_files[random_idx]
        else:
            img_file = self.img_files[idx]
        if os.path.isdir(self.data_dir):
            img_file = os.path.join(self.data_dir, img_file.split(",")[0])
        img_file = img_file.strip()
        strs = img_file.split("/")
        modality = strs[-4]
        sample_id = strs[-2]
        dataset_name = strs[-3]
        object_name = sample_id.split("_")[1]

        x = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        y = cv2.imread(img_file.replace("_img", "_mask"), cv2.IMREAD_GRAYSCALE)
        # if x is None or y is None or x.shape != y.shape:
        #     return self.__getitem__((idx + 1) % self.__len__())
        cropped_x, cropped_y = self._center_crop(x, y, min_crop_scale=self.min_crop_scale, max_crop_scale=self.max_crop_scale)
        x_augmented = self.transform(image=x, mask=y)
        support_x = x_augmented["image"] # (1, 224, 224)
        support_y = x_augmented["mask"].float() / 255.0 # (224, 224)
        query_xs = torch.stack([support_x,], dim=0) # (N_queries, 1, 224, 224)
        query_ys = torch.stack([support_y,], dim=0).float() # (N_queries, 224, 224)

        return {
            "modality": modality,
            "sample_id": sample_id,
            "support_x": support_x,
            "support_y": support_y,
            "query_xs": query_xs,
            "query_ys": query_ys,
            "dataset_name": dataset_name,
            "object_name": object_name,
        }


class FastBox2SegDataset(Dataset):
    def __init__(self, args, split_file, split, printer=print):
        super().__init__()
        self.split = split
        self.data_dir = args.data_dir
        # self.img_files = glob(os.path.join(split_file, "*", "*", "*", "*___img____*.jpg"))
        self.img_files = open(split_file, "r").readlines()
        self.train_iter_samples = args.train_iter_samples
        self.eval_iter_samples = args.eval_iter_samples
        self.len_samples = len(self.img_files)
        self.transform = A.Compose([
            A.Normalize(mean=[0.5], std=(0.5)), 
            ToTensorV2(),
        ])
        printer(f"[{self.split}] {split_file}: {self.len_samples} samples | sampling {self.__len__()} samples")

    def __len__(self):
        if self.split in ["train",] and self.train_iter_samples > 0:
            return self.train_iter_samples
        elif self.split not in ["train",] and self.eval_iter_samples > 0:
            return self.eval_iter_samples
        else:
            return len(self.samples)

    def __getitem__(self, index):
        if self.split in ["train",] and self.train_iter_samples > 0:
            random_idx = np.random.randint(0, self.len_samples)
            img_file = self.img_files[random_idx]
        elif self.split not in ["train",] and self.eval_iter_samples > 0:
            random_idx = np.random.randint(0, self.len_samples)
            img_file = self.img_files[random_idx]
        else:
            img_file = self.img_files[idx]
        img_file = os.path.join(self.data_dir, img_file.strip())
        mask_file = img_file.replace("___img____", "___mask____")
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        augmented = self.transform(image=img, mask=mask)
        support_x = augmented["image"]
        support_y = (augmented["mask"]).float() / 255.0
        query_xs = torch.stack([support_x,], dim=0) # (N_queries, 1, 224, 224)
        query_ys = torch.stack([support_y,], dim=0).float() # (N_queries, 224, 224)
        return {
            "support_x": support_x,
            "support_y": support_y,
            "query_xs": query_xs,
            "query_ys": query_ys,
        }
