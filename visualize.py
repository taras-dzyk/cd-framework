import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import os
from dataset import LevirCDDataset
from models.factory import get_model
from config import Config

def get_best_weights(model_name):
    model_dir = os.path.join(Config.CHECKPOINTS_DIR, model_name)
    if not os.path.exists(model_dir):
        return None
    files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not files:
        return None
   
    files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    return os.path.join(model_dir, files[-1])

def visualize():
    parser = argparse.ArgumentParser(description="CD Comparison Tool")
   
    parser.add_argument("--models", type=str, default="baseline,resnet18", 
                        help="Comma separated list of models (e.g. baseline,resnet18)")
    parser.add_argument("--indices", type=int, nargs='+', help="image indices")
    parser.add_argument("--num_samples", type=int, default=2, help="number of random pairs")
    args = parser.parse_args()

    
    model_names = [m.strip() for m in args.models.split(',')]
    
    
    loaded_models = {}
    for name in model_names:
        model = get_model(name).to(Config.DEVICE)
        weights_path = get_best_weights(name)
        if weights_path:
            model.load_state_dict(torch.load(weights_path, map_location=Config.DEVICE))
            model.eval()
            loaded_models[name] = model
            print(f"Loaded {name} weights: {weights_path}")
        else:
            print(f"Weights for {name} not found! Skipping...")

    # dataset
    test_ds = LevirCDDataset(Config.DATA_PATH, split='test', img_size=Config.IMAGE_SIZE)
    selected_indices = args.indices if args.indices else random.sample(range(len(test_ds)), args.num_samples)

    # grid
    num_cols = 3 + len(loaded_models)
    num_rows = len(selected_indices)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
    
    # oneline
    if num_rows == 1: axes = np.expand_dims(axes, axis=0)

    with torch.no_grad():
        for i, idx in enumerate(selected_indices):
            img_A, img_B, mask = test_ds[idx]
            
            # de-normallize
            def to_show(img):
                return img.permute(1, 2, 0).numpy() * 0.2 + 0.5

       
            axes[i, 0].imshow(to_show(img_A))
            axes[i, 0].set_title(f"T1 (idx {idx})")
            axes[i, 1].imshow(to_show(img_B))
            axes[i, 1].set_title("T2")
            axes[i, 2].imshow(mask.squeeze(), cmap='gray')
            axes[i, 2].set_title("Ground Truth")

          
            col_offset = 3
            in_A = img_A.unsqueeze(0).to(Config.DEVICE)
            in_B = img_B.unsqueeze(0).to(Config.DEVICE)
            
            for m_idx, (name, model) in enumerate(loaded_models.items()):
                pred = model(in_A, in_B).squeeze().cpu().numpy()
                pred_binary = (pred > 0.5).astype(np.uint8)
                
                axes[i, col_offset + m_idx].imshow(pred_binary, cmap='viridis')
                axes[i, col_offset + m_idx].set_title(f"Pred: {name}")

            for ax in axes[i]: ax.axis('off')

    plt.tight_layout()
    save_name = f"comparison_{'_'.join(loaded_models.keys())}.png"
    plt.savefig(save_name, dpi=150)
    print(f"Comparison saved as {save_name}")
    plt.show()

if __name__ == "__main__":
    visualize()