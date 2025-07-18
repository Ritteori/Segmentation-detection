from torch.utils.data import Dataset
import os
from torchvision import transforms
import cv2
import torch

class RSNADataset(Dataset):
    def __init__(self,images_dir,masks_dir,transform=None):
        super().__init__()
        
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        self.images_dirs = os.listdir(images_dir)
        
    def __len__(self):
        return len(self.images_dirs)
    
    def __getitem__(self, index):
        image_name = self.images_dirs[index]
        image_path = os.path.join(self.images_dir,image_name)
        mask_path = os.path.join(self.masks_dir,image_name)
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
        
        if self.transform is not None:
            augmented = self.transform(image=image,mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].float() / 255.0
            mask = mask.unsqueeze(0)
        else:
            image = transforms.ToTensor()(image)
            mask = torch.from_numpy(mask).float() / 255.0
            mask = mask.unsqueeze(0)
            
        return image,mask
        
        