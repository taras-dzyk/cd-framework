import torch
import os

class Config:
    DATA_PATH = "data/levir-cd"
    CHECKPOINT_DIR = "checkpoints"
    
    IMAGE_SIZE = 256
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    EPOCHS = 20
    
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")

    MODEL_NAME = "resnet18"

    @staticmethod
    def prepare_dirs():
        if not os.path.exists(Config.CHECKPOINTS_DIR):
            os.makedirs(Config.CHECKPOINTS_DIR)
