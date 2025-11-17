import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

def load_training_log(log_path='training_log.pkl'):
    """Load training log if it exists."""
    if not os.path.exists(log_path):
        print(f"No training log found at {log_path}")
        return None
    with open(log_path, 'rb') as f:
        return pickle.load(f)

def plot_training_curves(log_data, save_dir='visualizations'):
    """Plot and save training curves."""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(log_data['train_loss'], label='Train')
    if 'val_loss' in log_data:
        plt.plot(log_data['val_loss'], label='Validation')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(log_data['train_acc'], label='Train')
    if 'val_acc' in log_data:
        plt.plot(log_data['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_spike_raster(spikes, save_path='visualizations/spike_raster.png', n_neurons=100, n_timesteps=100):
    """Plot spike raster plot."""
    if not isinstance(spikes, list) or len(spikes) == 0:
        print("No spike data provided")
        return
        
    # Take first batch, first layer
    spikes = spikes[0][0].cpu().numpy()  # [timesteps, batch, neurons]
    spikes = spikes[:n_timesteps, 0, :n_neurons]  # Limit neurons and timesteps for visualization
    
    plt.figure(figsize=(12, 6))
    plt.eventplot([np.where(spikes[:, i])[0] for i in range(spikes.shape[1])], 
                 colors='k', lineoffsets=range(spikes.shape[1]), linelengths=0.8)
    plt.title('Spike Raster Plot')
    plt.xlabel('Timestep')
    plt.ylabel('Neuron Index')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_firing_rates(spikes, save_path='visualizations/firing_rates.png'):
    """Plot firing rate distribution across neurons."""
    if not isinstance(spikes, list) or len(spikes) == 0:
        print("No spike data provided")
        return
        
    # Calculate mean firing rate per neuron across batches and timesteps
    rates = [s[0].mean(dim=(0, 1)).cpu().numpy() for s in spikes]  # Mean over batch and time
    
    plt.figure(figsize=(10, 5))
    for i, layer_rates in enumerate(rates):
        plt.hist(layer_rates, bins=30, alpha=0.6, label=f'Layer {i+1}')
    
    plt.title('Firing Rate Distribution')
    plt.xlabel('Firing Rate (spikes/timestep)')
    plt.ylabel('Number of Neurons')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Create visualizations directory
    os.makedirs('visualizations', exist_ok=True)
    
    # Load training log if it exists
    log_data = load_training_log()
    if log_data:
        plot_training_curves(log_data)
    
    # Example usage with dummy data if no real spikes available
    # spikes = [torch.randn(10, 32, 256) > 0.9 for _ in range(2)]  # Dummy data
    # plot_spike_raster(spikes)
    # plot_firing_rates(spikes)
    
    print("Visualization scripts ready. Run after training to generate plots.")
