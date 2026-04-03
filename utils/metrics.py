import torch
import torch.nn as nn

#dice loss for binary segmentation
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_score

#combined BCE and Dice Loss for more robust and smoother gradients
class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()

    def forward(self, preds, targets):
        # Dice Loss Calculation
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score
        
        # BCE Loss Calculation
        bce_loss = self.bce(preds, targets)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

#helper to calculate precision, recall, and f1-score
def calculate_metrics(preds, targets):
    preds = (preds > 0.5).float()
    tp = (targets * preds).sum().item()
    fp = ((1 - targets) * preds).sum().item()
    fn = (targets * (1 - preds)).sum().item()
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    return precision, recall, f1