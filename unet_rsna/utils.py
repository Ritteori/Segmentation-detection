from tqdm import tqdm
import torch
import os

def calculate_mean_std(dataset,limit = 1000):
    
    total_mean = torch.tensor([0.0, 0.0, 0.0])
    total_std = torch.tensor([0.0, 0.0, 0.0])
    
    for i in tqdm(range(limit),desc='Calculating...'):
        
        image, _ = dataset[i]
        
        total_mean += image.mean(dim=(1, 2))
        total_std += image.std(dim=(1, 2))
        
    avg_mean = total_mean / limit
    avg_std = total_std / limit
    
    return avg_mean,avg_std
        
        