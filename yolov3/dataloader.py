import torch
from torch.utils.data import DataLoader
import albumentations as A
from dataset import YoloDataset
from config import LABELS_DIR,IMAGES_DIR,BATCH_SIZE

transforms = A.Compose([
    A.Resize(416,416),
    A.ToTensorV2(),

],bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

dataset = YoloDataset(LABELS_DIR,IMAGES_DIR,transforms)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x))
)