"""Visualization utilities for ChaosNet."""
from chaosnet.visualize_spikes import plot_spike_raster, plot_population_activity
from chaosnet.visualize_firing_rates import plot_firing_rates, plot_histogram
from chaosnet.visualize_ei_balance import plot_ei_balance
from chaosnet.visualize_membrane import plot_membrane_potential
from chaosnet.visualize_activity_gif import create_activity_animation

__all__ = [
    'plot_spike_raster',
    'plot_population_activity',
    'plot_firing_rates',
    'plot_histogram',
    'plot_ei_balance',
    'plot_membrane_potential',
    'create_activity_animation'
]
