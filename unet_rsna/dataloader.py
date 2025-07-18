from torch.utils.data import DataLoader,random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import RSNADataset
from config import IMAGES_DIR,MASKS_DIR,MEAN,STD,BATCH_SIZE


transforms = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.3),
    A.ShiftScaleRotate(
        shift_limit=0.02,
        scale_limit=0.05,
        rotate_limit=3,
        p=0.3
    ),
    A.GaussianBlur(p=0.2),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])


dataset = RSNADataset(IMAGES_DIR,MASKS_DIR,transforms)

dataset_size = len(dataset)
val_size = int(0.2 * dataset_size)
train_size = dataset_size - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)



