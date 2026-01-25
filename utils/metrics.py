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

def calculate_iou(preds, targets, smooth=1e-6):
    #converting probs to 0 or 1
    preds = (preds > 0.5).float()
    
    #intersection is where both are 1
    intersection = (preds * targets).sum()
    
    #union is area of both combined
    total = (preds + targets).sum()
    union = total - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()
