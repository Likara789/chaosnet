"""Visualization of membrane potential dynamics in ChaosNet."""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Tuple
import torch
import os

def plot_membrane_potential(
    potentials: Union[torch.Tensor, np.ndarray],
    spikes: Optional[Union[torch.Tensor, np.ndarray]] = None,
    n_neurons: int = 5,
    n_timesteps: int = 200,
    threshold: float = 0.3,
    figsize: tuple = (14, 6),
    title: str = 'Membrane Potential Dynamics',
    save_path: Optional[str] = 'visualizations/membrane_potential.png',
    show: bool = False
) -> None:
    """Plot membrane potential traces for sample neurons.
    
    Args:
        potentials: Tensor of shape (timesteps, batch, neurons) or (timesteps, neurons)
        spikes: Optional spike tensor for marking spike times
        n_neurons: Number of example neurons to plot
        n_timesteps: Number of timesteps to display
        threshold: Spiking threshold (dashed line)
        figsize: Figure size (width, height)
        title: Plot title
        save_path: Path to save the figure (None to not save)
        show: Whether to display the plot
    """
    if isinstance(potentials, torch.Tensor):
        potentials = potentials.cpu().numpy()
    if isinstance(spikes, torch.Tensor):
        spikes = spikes.cpu().numpy()
    
    # Handle different input shapes
    if len(potentials.shape) == 2:  # (timesteps, neurons)
        potentials = np.expand_dims(potentials, axis=1)  # Add batch dim
    
    # Take first batch and limit timesteps
    potentials = potentials[:n_timesteps, 0, :n_neurons]
    
    if spikes is not None and len(spikes.shape) == 3:  # (timesteps, batch, neurons)
        spikes = spikes[:n_timesteps, 0, :n_neurons]
    else:
        spikes = None
    
    plt.figure(figsize=figsize)
    
    # Create time axis
    time = np.arange(potentials.shape[0])
    
    # Plot membrane potential for each neuron
    for n in range(potentials.shape[1]):
        plt.plot(time, potentials[:, n], label=f'Neuron {n+1}', alpha=0.8, linewidth=1.5)
        
        # Mark spikes if provided
        if spikes is not None:
            spike_times = np.where(spikes[:, n] > 0)[0]
            spike_vals = [potentials[t, n] for t in spike_times]
            plt.scatter(spike_times, spike_vals, color='red', s=30, zorder=5)
    
    # Add threshold line
    plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label='Threshold')
    
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('Membrane Potential', fontsize=12)
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    plt.close()

def plot_membrane_distribution(
    potentials: Union[torch.Tensor, np.ndarray],
    bins: int = 50,
    figsize: tuple = (10, 5),
    title: str = 'Membrane Potential Distribution',
    save_path: Optional[str] = 'visualizations/membrane_dist.png',
    show: bool = False
) -> None:
    """Plot distribution of membrane potentials across all neurons and timesteps.
    
    Args:
        potentials: Tensor of shape (timesteps, batch, neurons) or (timesteps, neurons)
        bins: Number of histogram bins
        figsize: Figure size (width, height)
        title: Plot title
        save_path: Path to save the figure (None to not save)
        show: Whether to display the plot
    """
    if isinstance(potentials, torch.Tensor):
        potentials = potentials.cpu().numpy()
    
    # Flatten all dimensions except the last (neuron dimension)
    if len(potentials.shape) == 3:  # (timesteps, batch, neurons)
        flat_potentials = potentials.reshape(-1, potentials.shape[-1])
    else:  # (timesteps, neurons)
        flat_potentials = potentials.reshape(-1)
    
    plt.figure(figsize=figsize)
    
    # Plot histogram
    plt.hist(flat_potentials, bins=bins, alpha=0.7, density=True)
    
    # Add vertical line at 0
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Resting Potential')
    
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel('Membrane Potential', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    plt.close()

# Example usage:
if __name__ == "__main__":
    import torch
    
    # Create sample data
    timesteps, batch_size, n_neurons = 200, 1, 5
    
    # Generate membrane potentials with some dynamics
    t = torch.linspace(0, 10, timesteps)
    potentials = torch.zeros(timesteps, batch_size, n_neurons)
    
    # Different frequency sine waves for each neuron
    for n in range(n_neurons):
        freq = 0.5 + n * 0.2
        noise = torch.randn(timesteps) * 0.1
        potentials[:, 0, n] = 0.5 * torch.sin(freq * t) + 0.1 * noise
    
    # Generate some spikes
    spikes = (torch.rand(timesteps, batch_size, n_neurons) > 0.98).float()
    
    # Create visualizations directory
    os.makedirs('visualizations', exist_ok=True)
    
    # Generate plots
    plot_membrane_potential(
        potentials, 
        spikes=spikes,
        save_path='visualizations/membrane_potential.png', 
        show=True
    )
    
    plot_membrane_distribution(
        potentials,
        save_path='visualizations/membrane_dist.png',
        show=True
    )
