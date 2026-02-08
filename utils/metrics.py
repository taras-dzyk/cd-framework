import torch

def calculate_metrics(pred, target):
    pred = (pred > 0.5).float()
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    
    return {
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "iou": iou.item()
    }