import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_history():
    parser = argparse.ArgumentParser(description="Compare training data")
    parser.add_argument("--models", type=str, default="baseline,resnet18", 
                        help="comma separated list of models (e.g. baseline,resnet18)")
    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(',')]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',"#bf11d6" ]
    markers = ['o', 's', '^', 'v']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for i, name in enumerate(model_names):
        path = f"checkpoints/{name}/history.csv"
        if not os.path.exists(path):
            print(f"File {name} not found: {path}")
            continue

        df = pd.read_csv(path)
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        # Loss 
        ax1.plot(df['epoch'], df['loss'], label=f'{name} Loss', 
                 color=color, marker=marker, linestyle='--', alpha=0.8)
        
        #IoU 
        ax2.plot(df['epoch'], df['iou'], label=f'{name} IoU', 
                 color=color, marker=marker, linewidth=2)
        
      
        if 'f1' in df.columns:
            ax2.plot(df['epoch'], df['f1'], label=f'{name} F1', 
                     color=color, marker=marker, linestyle=':', alpha=0.5)

    
    ax1.set_title('Збіжність функції втрат (Loss)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Епоха', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    ax2.set_title('Динаміка IoU та F1-Score', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Епоха', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    
   
    save_path = f"checkpoints/comparison_{'_'.join(model_names)}.png"
    plt.savefig(save_path, dpi=300)
    print(f"Comparison plot saved: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_history()