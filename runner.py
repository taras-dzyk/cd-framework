import argparse
import torch
from torch.utils.data import DataLoader
from config import Config
from dataset import LevirCDDataset
from models.factory import get_model
from utils.trainer import CDTrainer
import os

def main():
    parser = argparse.ArgumentParser(description="Master CD Framework")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], required=True)
    parser.add_argument("--model", type=str, default="baseline")
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS)
    parser.add_argument("--weights", type=str, default=None)
    args = parser.parse_args()


    Config.prepare_dirs()
     # dir for model
    model_save_dir = os.path.join(Config.CHECKPOINTS_DIR, args.model)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)


    print(f"Device: {Config.DEVICE}")
    # models
    model = get_model(args.model).to(Config.DEVICE)
    

    if args.mode == "eval" and args.weights is None:
        model_files = [f for f in os.listdir(model_save_dir) if f.endswith('.pth')]
        if model_files:
            model_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            args.weights = os.path.join(model_save_dir, model_files[-1])
            print(f"Automatically selected weights {args.weights}")
        else:
            print(f"Cannot find weights for {args.model} in {model_save_dir}")
            return

    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location=Config.DEVICE))
        print(f"Weights from {args.weights}")
    
    trainer = CDTrainer(model, Config, Config.DEVICE, args.model)

    if args.mode == "train":
        train_ds = LevirCDDataset(Config.DATA_PATH, split='train', img_size=Config.IMAGE_SIZE)
        val_ds = LevirCDDataset(Config.DATA_PATH, split='test', img_size=Config.IMAGE_SIZE)
        
        train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE)
        
        print(f"Begining training {args.model} on {args.epochs} epochs...")
        
        for epoch in range(1, args.epochs + 1):
            train_loss = trainer.train_epoch(train_loader, epoch)
            
            val_metrics = trainer.evaluate(val_loader)
            
            trainer.log_metrics(epoch, train_loss, val_metrics)
            
            print(f"Epoch {epoch}/{args.epochs} done. Loss: {train_loss:.4f} | IoU: {val_metrics['iou']:.4f} | F1: {val_metrics['f1']:.4f}")
            
            if epoch % 5 == 0 or epoch == args.epochs:
                save_path = os.path.join(model_save_dir, f"epoch_{epoch}.pth")
                torch.save(model.state_dict(), save_path)
                print(f"Model saved to {save_path}")


    elif args.mode == "eval":
        test_ds = LevirCDDataset(Config.DATA_PATH, split='test', img_size=Config.IMAGE_SIZE)
        test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE)
        
        print(f"Evaluation of {args.model}...")
        res = trainer.evaluate(test_loader)
        
        print(f"\n--- RESULTS {args.model.upper()} ---")
        for k, v in res.items():
            print(f"{k.upper():<10}: {v:.4f}")

if __name__ == "__main__":
    main()