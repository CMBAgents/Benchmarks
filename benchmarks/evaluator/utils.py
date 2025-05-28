import numpy as np
import matplotlib.pyplot as plt

def plot_success_bars(data_arrays, labels, width=0.25):
    """
    Plots grouped bar chart for multiple data arrays.

    Parameters:
    - data_arrays: list of np.arrays or lists, each representing a success metric.
    - labels: list of strings, legend labels for each array.
    - width: float, width of each bar.
    """
    x = np.arange(len(data_arrays[0]))
    total_cases = len(data_arrays)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (data, label) in enumerate(zip(data_arrays, labels)):
        ax.bar(x + (i - total_cases // 2) * width + width / 2, np.array(data) + 0.1,
               width, bottom=-0.1, label=label, zorder=3)

    ax.set_ylabel('Success Rate')
    ax.set_xlabel('Problem Index')
    ax.set_xticks(x)
    ax.set_ylim(-0.1, 1.1)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6, zorder=1)
    plt.tight_layout()
    plt.show()