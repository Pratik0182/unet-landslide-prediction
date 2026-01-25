import os
import h5py
import numpy as np
import torch
import random
from torch.utils.data import Dataset

#custom dataset class for loading landslide images and corresponding masks
class LandslideDataset(Dataset):
    def __init__(self, image_dir, mask_dir, filenames_file, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augment = augment
        
        #load filenames from the .txt file
        with open(filenames_file, "r") as f:
            self.filenames = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.filenames)


    def __getitem__(self, index):
        f_name = self.filenames[index]
        image_path = os.path.join(self.image_dir, f_name)
        mask_path = os.path.join(self.mask_dir, f_name)
        
        #load hdf5 image and mask
        with h5py.File(image_path, "r") as f:
            image = f["img"][:]

        with h5py.File(mask_path, "r") as f:
            mask = f["mask"][:]


        #normalize image to 0–1 range
        min_val, max_val = image.min(), image.max()
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val)
        else:
            image = np.zeros_like(image)
            
        """
        #old normalization logic, kept it just in case simple min-max fails
        #mean = np.mean(image, axis=(0, 1))
        #std = np.std(image, axis=(0, 1))
        #image = (image - mean) / (std + 1e-8)
        """

        #augmenting data if flag is set to true
        if self.augment:
            
            #horizontal flip
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()

            #vertical flip 
            if random.random() > 0.5:
                image = np.flipud(image).copy()
                mask = np.flipud(mask).copy()

            #random rotation 90 degrees
            if random.random() > 0.5:
                k = random.randint(1, 3) 
                image = np.rot90(image, k).copy()
                mask = np.rot90(mask, k).copy()


        #convert to tensors and adjust dimensions
        image = torch.from_numpy(image).float().permute(2, 0, 1)  # (C, H, W)
        mask = torch.from_numpy(mask).float().unsqueeze(0)         # (1, H, W)
        
        return image, mask