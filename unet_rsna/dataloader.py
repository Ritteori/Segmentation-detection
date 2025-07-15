from torch.utils.data import DataLoader,random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import RSNADataset
from config import IMAGES_DIR,MASKS_DIR,MEAN,STD,BATCH_SIZE

if __name__ == 'main':

    transforms = A.Compose(
        [
            A.HorizontalFlip(0.5),
            A.Resize(256,256),
            A.ShiftScaleRotate(0.5),
            A.Normalize(mean=MEAN,std=STD),
            ToTensorV2()
        ]
    )

    dataset = RSNADataset(IMAGES_DIR,MASKS_DIR,transforms)

    dataset_size = len(dataset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=-1, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=-1, pin_memory=True)



