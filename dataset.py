import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms

class LevirCDDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=256):
        self.root_dir = os.path.join(root_dir, split)
        self.dir_A = os.path.join(self.root_dir, 'A')
        self.dir_B = os.path.join(self.root_dir, 'B')
        self.dir_label = os.path.join(self.root_dir, 'label')
        
        # workaround for .DS_Store)
        self.file_names = sorted([
            f for f in os.listdir(self.dir_A) 
            if not f.startswith('.') and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))
        ])
        
        #  ImageNet normalization
        self.transform_img = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        

        self.transform_label = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        name = self.file_names[idx]
        
        img_A = Image.open(os.path.join(self.dir_A, name)).convert('RGB')
        img_B = Image.open(os.path.join(self.dir_B, name)).convert('RGB')
        label = Image.open(os.path.join(self.dir_label, name)).convert('L')
        
        img_A = self.transform_img(img_A)
        img_B = self.transform_img(img_B)
        label = self.transform_label(label)
        
        # trans to 0 or 1
        label = (label > 0.5).float()
        
        return img_A, img_B, label