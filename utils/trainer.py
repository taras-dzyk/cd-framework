import torch
import csv
import os
import numpy as np
from tqdm import tqdm
from utils.metrics import calculate_metrics

class CDTrainer:
    def __init__(self, model, config, device, model_name):
        self.model = model
        self.config = config
        self.device = device
        self.model_name = model_name
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        self.criterion = torch.nn.BCELoss()
        
    
        self.log_path = os.path.join(config.CHECKPOINTS_DIR, model_name, "history.csv")
        
   
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "loss", "precision", "recall", "f1", "iou"])

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
        
        for img_A, img_B, mask in loop:
            img_A, img_B, mask = img_A.to(self.device), img_B.to(self.device), mask.to(self.device)
            
            self.optimizer.zero_grad()
            pred = self.model(img_A, img_B)
            loss = self.criterion(pred, mask)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")
        
        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        metrics_list = []
        
        with torch.no_grad():
            for img_A, img_B, mask in tqdm(dataloader, desc="Evaluating", leave=False):
                img_A, img_B, mask = img_A.to(self.device), img_B.to(self.device), mask.to(self.device)
                pred = self.model(img_A, img_B)
                m = calculate_metrics(pred, mask)
                metrics_list.append(list(m.values()))
        
    
        avg_metrics = np.mean(metrics_list, axis=0)
        return dict(zip(["precision", "recall", "f1", "iou"], avg_metrics))

    def log_metrics(self, epoch, loss, metrics_dict):

        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, 
                f"{loss:.4f}", 
                f"{metrics_dict['precision']:.4f}",
                f"{metrics_dict['recall']:.4f}",
                f"{metrics_dict['f1']:.4f}",
                f"{metrics_dict['iou']:.4f}"
            ])