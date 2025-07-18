from tqdm import tqdm
import torch
import os
from tqdm import tqdm
import csv
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib import pyplot as plt

from config import MODEL_PATH,LOG_PATH

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
        
def iou_score(preds, targets, threshold=0.5, eps=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    targets = targets.float()
    
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = ((preds + targets) >= 1).sum(dim=(1, 2, 3))
    
    iou = (intersection + eps) / (union + eps)
    return iou.mean()

def dice_score(preds, targets, threshold=0.5, eps=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    targets = targets.float()
    
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    total = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    
    dice = (2 * intersection + eps) / (total + eps)
    return dice.mean()

def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, scaler):
    
    epoch_loss = 0.0
    epoch_iou = 0.0
    epoch_dice = 0.0
    
    for real_image, mask in tqdm(dataloader,desc='Training...'):
        
        optimizer.zero_grad()
        
        real_image = real_image.to(device)
        mask = mask.to(device)
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, cache_enabled=True):
            preds = model(real_image)
            loss = loss_fn(preds.float(), mask.float())

        epoch_loss += loss.item()
    
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        with torch.no_grad():
            epoch_iou += iou_score(preds,mask).item()
            epoch_dice += dice_score(preds,mask).item()
            
    epoch_loss /= len(dataloader)
    epoch_iou /= len(dataloader)
    epoch_dice /= len(dataloader)
    
    print(f'Epoch:{epoch + 1} | Loss:{epoch_loss:.4f} | Iou:{epoch_iou:.4f} | Dice:{epoch_dice:.4f}')

def eval_epoch(model, dataloader, loss_fn, device, epoch):
    model.eval()
    
    epoch_loss = 0.0
    epoch_iou = 0.0
    epoch_dice = 0.0
    
    with torch.no_grad():
        for real_image, mask in tqdm(dataloader,desc='Evaluating...'):
            real_image = real_image.to(device)
            mask = mask.to(device)
            
            preds = model(real_image)
            loss = loss_fn(preds,mask)
            epoch_loss += loss.item()
            
            epoch_iou += iou_score(preds,mask).item()
            epoch_dice += dice_score(preds,mask).item()
            
        epoch_loss /= len(dataloader)
        epoch_iou /= len(dataloader)
        epoch_dice /= len(dataloader)
        
        print(f'Epoch:{epoch + 1} | EvalLoss:{epoch_loss:.4f} | EvalIou:{epoch_iou:.4f} | EvalDice:{epoch_dice:.4f}')
        
    model.train()    
    
    return epoch + 1,epoch_loss,epoch_iou,epoch_dice
            
def train_loop(model, train_loader, val_loader, optimizer, loss_fn, device, epochs, log_path, log_every=1):
    model.train()
    
    scheduler = ReduceLROnPlateau(optimizer,mode='max',factor=0.5,patience=3,verbose=True)
    
    model = model.to(device)
    scaler = torch.amp.GradScaler()
    
    file_exists = os.path.isfile(log_path)
    mode = 'a' if file_exists else 'w'
    
    with open(log_path, mode, newline='', encoding='utf-8') as file:
        fieldnames = ['Epoch', 'EvalLoss', 'EvalIoU', 'EvalDice']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
    
        for epoch in range(epochs):
            train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, scaler)
            
            if (epoch + 1) % log_every == 0:
                epoch_, loss_, iou_, dice_ = eval_epoch(model, val_loader, loss_fn, device, epoch)
                writer.writerow({'Epoch': epoch_, 'EvalLoss': loss_, 'EvalIoU': iou_, 'EvalDice': dice_})
                
            if (epoch + 1) % 6 == 0:
                
                os.makedirs(MODEL_PATH, exist_ok=True)
                
                checkpoint_path = os.path.join(MODEL_PATH,f'unet_{epoch + 1}epochs.pth')
                torch.save({
                    'epoch': epoch_,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss_,
                    'iou': iou_,
                    'dice': dice_
                }, checkpoint_path)
                
            scheduler.step(dice_)

def calculate_pos_weight(loader):
    positive = 0
    total = 0
    for _, mask in loader:
        positive += mask.sum()
        total += mask.numel()
    return (total - positive) / positive

def show_result(val_dataloader, model, device, max_images=8, threshold=0.5):
    
    model.eval()
    images_shown = 0
    plt.figure(figsize=(16,max_images * 2))
    
    for batch in val_dataloader:
        images, masks = batch
        images = images.to(device)
        
        logits = model(images)
        preds = torch.sigmoid(logits)
        preds = (preds > threshold).float()
        
        for i in range(16):
            if images_shown >= max_images:
                break
            
            img = images[i][0].cpu().numpy() # (H,W)
            mask = masks[i][0].cpu().numpy()
            pred = preds[i][0].cpu().numpy()
            
            img = (img - img.min()) / (img.max() - img.min())
            
            plt.subplot(max_images,2,images_shown * 2 + 1)
            plt.imshow(img, cmap='gray')
            plt.imshow(mask, cmap='Reds', alpha=0.4)
            plt.title("Ground Truth")
            plt.axis(False)
            
            plt.subplot(max_images,2,images_shown * 2 + 2)
            plt.imshow(img,cmap="gray")
            plt.imshow(pred, cmap='Reds', alpha=0.4)
            plt.title("Prediction")
            plt.axis(False)
            
            images_shown += 1
            
        if images_shown >= max_images:
            break
            
        
    plt.tight_layout()
    plt.show()
    model.train()
            
def eval_segmentation_metrics(val_dataloader, model, device, threshold=0.5):
    model.eval()
    
    TP = 0
    FP = 0
    FN = 0

    with torch.no_grad():
        for images, masks in tqdm(val_dataloader,desc='Estimating...'):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            preds = torch.sigmoid(logits)
            preds = (preds > threshold).float()

            TP += ((preds == 1) & (masks == 1)).sum().item()
            FP += ((preds == 1) & (masks == 0)).sum().item()
            FN += ((preds == 0) & (masks == 1)).sum().item()
    
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    
    model.train()

def evaluate_metrics():
    
    with open(LOG_PATH,"r") as f:
        reader = csv.DictReader(f)
        
        epochs, eval_loss, eval_iou, eval_dice = [], [], [], []
        
        for row in reader:
            epochs.append(int(row['Epoch']))
            eval_loss.append(float(row['EvalLoss']))
            eval_iou.append(float(row['EvalIoU']))
            eval_dice.append(float(row['EvalDice']))
        
        plt.figure(figsize=(7,5))
        plt.plot(epochs, eval_loss, marker='.', linestyle='-', color='blue', label='Loss')
        plt.plot(epochs, eval_iou, marker='o', linestyle='-', color='red', label='IoU')
        plt.plot(epochs, eval_dice, marker='x', linestyle='-', color='green', label='Dice')
        
        plt.legend()
        
        plt.tight_layout()
        plt.show()