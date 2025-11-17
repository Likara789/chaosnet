"""Generate visualizations for ChaosNet training results."""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Union

def load_training_log(exp_dir: Union[str, Path]) -> Dict:
    """Load training log from experiment directory."""
    log_path = Path(exp_dir) / "training_log.pkl"
    with open(log_path, 'rb') as f:
        return pickle.load(f)

def plot_training_curves(
    log_data: Dict,
    save_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """Plot training and validation curves."""
    plt.figure(figsize=(14, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(log_data['train_losses'], label='Train')
    if 'val_losses' in log_data and log_data['val_losses']:
        plt.plot(log_data['val_losses'], label='Validation')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(log_data['train_accs'], label='Train')
    if 'val_accs' in log_data and log_data['val_accs']:
        plt.plot(log_data['val_accs'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    plt.close()

def visualize_experiment(
    exp_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None
) -> None:
    """Generate all visualizations for an experiment."""
    exp_dir = Path(exp_dir)
    if output_dir is None:
        output_dir = exp_dir / "visualizations"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating visualizations for experiment: {exp_dir.name}")
    
    # 1. Load training log
    log_data = load_training_log(exp_dir)
    
    # 2. Plot training curves
    print("Generating training curves...")
    plot_training_curves(
        log_data,
        save_path=output_dir / 'training_curves.png',
        show=False
    )
    
    # 3. Generate model-specific visualizations
    # (This would be specific to your model architecture)
    # visualize_model(exp_dir, output_dir)
    
    print(f"\nVisualizations saved to: {output_dir.absolute()}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate visualizations for ChaosNet experiments')
    parser.add_argument('--exp-dir', type=str, default='experiments/latest',
                       help='Path to experiment directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for visualizations (default: <exp_dir>/visualizations)')
    
    args = parser.parse_args()
    
    # If no experiment directory is specified, find the most recent one
    if args.exp_dir == 'experiments/latest':
        exp_dirs = sorted(Path('experiments').glob('exp_*'), key=os.path.getmtime, reverse=True)
        if exp_dirs:
            args.exp_dir = exp_dirs[0]
        else:
            print("No experiment directories found in 'experiments/'")
            return
    
    visualize_experiment(args.exp_dir, args.output_dir)

if __name__ == "__main__":
    main()
