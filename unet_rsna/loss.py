import torch
from torch import nn

class DiceLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        smooth = 1e-6
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        return 1 - (2. * intersection + smooth) / (union + smooth)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', eps=1e-8, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        
    def forward(self, logit, target):
        pred = torch.sigmoid(logit)
        
        p_t = pred * target + (1 - pred) * (1 - target)
        log_p_t = torch.log(p_t + self.eps)
        weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        focal_loss = -alpha_t * weight * log_p_t
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
        
class TripleLoss(nn.Module):
    def __init__(self,  alpha=0.7, beta = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = alpha
        self.beta = beta

        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, pred, target):
        
        dice = self.dice(pred, target)
        focal= self.focal(pred,target)
        bce = self.bce(pred, target)
        
        return self.alpha * dice + self.beta * bce + (1 - self.alpha - self.beta) * focal
