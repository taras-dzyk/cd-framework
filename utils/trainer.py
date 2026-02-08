import torch
from tqdm import tqdm
from utils.metrics import calculate_metrics
import os

class CDTrainer:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        self.criterion = torch.nn.BCELoss()

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        loop = tqdm(dataloader, desc="Training")
        for img_A, img_B, mask in loop:
            img_A, img_B, mask = img_A.to(self.device), img_B.to(self.device), mask.to(self.device)
            
            self.optimizer.zero_grad()
            pred = self.model(img_A, img_B)
            loss = self.criterion(pred, mask)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        metrics_list = []
        with torch.no_grad():
            for img_A, img_B, mask in tqdm(dataloader, desc="Evaluating"):
                img_A, img_B, mask = img_A.to(self.device), img_B.to(self.device), mask.to(self.device)
                pred = self.model(img_A, img_B)
                m = calculate_metrics(pred, mask)
                metrics_list.append(list(m.values()))
        
        # Середнє по всіх метриках
        import numpy as np
        avg = np.mean(metrics_list, axis=0)
        return dict(zip(["precision", "recall", "f1", "iou"], avg))