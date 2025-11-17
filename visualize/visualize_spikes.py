"""Visualization functions for spike data in ChaosNet."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from typing import List, Optional, Union
import torch

def plot_spike_raster(
    spikes: Union[torch.Tensor, np.ndarray],
    n_neurons: int = 100,
    n_timesteps: int = 100,
    figsize: tuple = (12, 6),
    title: str = 'Spike Raster Plot',
    save_path: Optional[str] = 'visualizations/spike_raster.png',
    show: bool = False
) -> None:
    """Plot a spike raster plot.
    
    Args:
        spikes: Tensor or array of shape (timesteps, batch, neurons)
        n_neurons: Number of neurons to display (top n)
        n_timesteps: Number of timesteps to display
        figsize: Figure size (width, height)
        title: Plot title
        save_path: Path to save the figure (None to not save)
        show: Whether to display the plot
    """
    if isinstance(spikes, torch.Tensor):
        spikes = spikes.cpu().numpy()
    
    # Limit neurons and timesteps for visualization
    spikes = spikes[:n_timesteps, 0, :n_neurons]  # Take first batch
    
    plt.figure(figsize=figsize)
    
    # Create spike trains for each neuron
    spike_times = [np.where(spikes[:, i])[0] for i in range(spikes.shape[1])]
    
    # Plot spikes
    plt.eventplot(
        spike_times,
        colors='black',
        lineoffsets=range(len(spike_times)),
        linelengths=0.8,
        linewidths=0.8
    )
    
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('Neuron Index', fontsize=12)
    plt.yticks([0, n_neurons-1], [0, n_neurons-1])
    plt.grid(True, alpha=0.2, linestyle='--')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    plt.close()

def plot_population_activity(
    spikes: Union[torch.Tensor, np.ndarray],
    window_size: int = 10,
    figsize: tuple = (12, 4),
    title: str = 'Population Activity',
    save_path: Optional[str] = 'visualizations/population_activity.png',
    show: bool = False
) -> None:
    """Plot population firing rate over time.
    
    Args:
        spikes: Tensor or array of shape (timesteps, batch, neurons)
        window_size: Size of sliding window for smoothing
        figsize: Figure size (width, height)
        title: Plot title
        save_path: Path to save the figure (None to not save)
        show: Whether to display the plot
    """
    if isinstance(spikes, torch.Tensor):
        spikes = spikes.cpu().numpy()
    
    # Sum over neurons and average over batch
    pop_activity = spikes.mean(axis=2)  # (timesteps, batch)
    
    # Smooth with sliding window
    kernel = np.ones(window_size) / window_size
    smoothed = np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode='same'),
        axis=0,
        arr=pop_activity
    )
    
    plt.figure(figsize=figsize)
    
    # Plot each batch separately with transparency
    for b in range(smoothed.shape[1]):
        plt.plot(smoothed[:, b], alpha=0.6, linewidth=1.5, 
                label=f'Batch {b+1}' if b < 5 else None)
    
    if smoothed.shape[1] > 5:
        plt.plot([], [], ' ', label=f'+{smoothed.shape[1]-5} more batches')
    
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('Population Firing Rate', fontsize=12)
    plt.grid(True, alpha=0.2, linestyle='--')
    
    if smoothed.shape[1] <= 5:  # Only show legend if few batches
        plt.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    plt.close()

# Example usage:
if __name__ == "__main__":
    import os
    import torch
    
    # Create sample data
    timesteps, batch_size, n_neurons = 200, 1, 100
    spikes = (torch.rand(timesteps, batch_size, n_neurons) > 0.95).float()
    
    # Create visualizations directory
    os.makedirs('visualizations', exist_ok=True)
    
    # Generate plots
    plot_spike_raster(spikes, save_path='visualizations/spike_raster.png', show=True)
    plot_population_activity(spikes, save_path='visualizations/population_activity.png', show=True)
