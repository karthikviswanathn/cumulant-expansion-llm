import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def plot_stats(stats, ax, title = None, xarr = None, xlabel = None):
    """
    Plot normalized cumulants using specified colorblind-friendly palette.
    Args:
        stats: Dictionary containing analysis statistics
        ax: Matplotlib axis to plot on
        shuffled: Whether this is for shuffled data
    """
    # Use the specified colorblind-friendly palette
    hex6 = ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000', '#33A02C', '#FB9A99']
    colors6 = [mcolors.to_rgb(i) for i in hex6]
    
    # Set color cycle for the axis
    ax.set_prop_cycle('color', colors6)
    
    # Get cumulants data
    cumulants = stats['avg_normalized_cumulants']
    
    
    # Plot each cumulant
    for idx in range(7):
        if xarr is None: xarr = np.linspace(0, 1, cumulants.shape[0])
        ax.plot(xarr,
            cumulants.T[idx],
            marker='.',
            label=f"$\\kappa_{{{idx+2}}}$"
        )
    
    # Set title and add grid
    if title is not None: ax.set_title(title, fontsize = "x-large")
    ax.grid(True)
    if xlabel is None: xlabel = 'Relative Depth'
    ax.set_xlabel(xlabel, fontsize = "x-large")
    ax.set_ylabel('Normalized Cumulant Value', fontsize = "x-large")
    
    # Add y=0 reference line
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    ax.tick_params(which='both', labelsize="x-large")
    

def plot_comparison(stats, shuffled_stats, figsize=(14, 4)):
    """
    Create a three-panel figure comparing structured and shuffled results.
    
    Args:
        stats: Dictionary of statistics for structured data
        shuffled_stats: Dictionary of statistics for shuffled data
        figsize: Figure size tuple
    """
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Panel 1: Structured cumulants
    plot_stats(stats, axes[0], title = "Structured")
    
    # Panel 2: Shuffled cumulants
    plot_stats(shuffled_stats, axes[1], title = "Shuffled")
    
    # Panel 3: Entropy using blue and orange
    ax_entropy = axes[2]
    blue = "#0072B2"    # Blue for structured
    orange = "#E69F00"  # Orange for shuffled
    num_entries = stats['entropy_com'].shape[0]
    ax_entropy.plot(
        np.linspace(0, 1, num_entries),
        stats['entropy_com'],
        linestyle='dotted',
        color=blue,
        label=r"$S_{\mathrm{structured}}(\mu)$"
    )
    ax_entropy.plot(
        np.linspace(0, 1, num_entries),
        stats['avg_entropy'],
        marker='.',
        color=blue,
        label=r"$\langle S_{\mathrm{structured}} \rangle$"
    )
    ax_entropy.plot(
        np.linspace(0, 1, num_entries),
        shuffled_stats['entropy_com'],
        linestyle='dotted',
        color=orange,
        label=r"$S_{\mathrm{shuffled}}(\mu)$"
    )
    ax_entropy.plot(
        np.linspace(0, 1, num_entries),
        shuffled_stats['avg_entropy'],
        marker='.',
        color=orange,
        label=r"$\langle S_{\mathrm{shuffled}} \rangle$"
    )
    
    ax_entropy.set_title("Entropy", fontsize = "x-large")
    ax_entropy.grid(True)
    ax_entropy.set_xlabel('Relative Depth', fontsize = "x-large")
    ax_entropy.set_ylabel('Entropy', fontsize = "x-large")
    # ax_entropy.legend(loc='best', fontsize = "large")
    
    ax_entropy.tick_params(which='both', labelsize="x-large")
    
    # Collect handles and labels separately for shuffled/structured cumulants and entropy
    handles_labels_0 = axes[0].get_legend_handles_labels()  # Structured cumulants
    handles_labels_2 = axes[2].get_legend_handles_labels()  # Entropy
    
    # Legend line 1: Cumulants
    fig.legend(
        handles_labels_0[0],
        handles_labels_0[1],
        loc='upper center',
        bbox_to_anchor=(0.5, -0.02),  # Line 1
        ncol=len(handles_labels_0[0]),
        fontsize="x-large"
    )
    
    # Legend line 2: Entropy
    fig.legend(
        handles_labels_2[0],
        handles_labels_2[1],
        loc='upper center',
        bbox_to_anchor=(0.5, -0.10),  # Line 2 (lower than line 1)
        ncol=len(handles_labels_2[0]),
        fontsize="x-large"
    )
    
    # Adjust layout
    plt.tight_layout()
    
    return fig
