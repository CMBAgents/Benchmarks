# filename: codebase/analyze_power_spectrum_results.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time


def analyze_power_spectrum_results():
    """
    Analyzes and visualizes the relative difference in the matter power spectrum
    stored in 'result.csv'.
    Checks the CSV file, calculates statistics, and generates a plot.
    """
    plt.rcParams['text.usetex'] = False

    # Define file paths
    output_dir = "data"
    csv_file_path = os.path.join(output_dir, "result.csv")

    # --- CSV File Check ---
    print("--- CSV File Check (" + csv_file_path + ") ---")
    if not os.path.exists(csv_file_path):
        print("Error: CSV file not found at " + csv_file_path)
        return

    try:
        df = pd.read_csv(csv_file_path)
        print("File exists: Yes")
        print("Readable: Yes")
    except Exception as e:
        print("Error: Could not read CSV file.")
        print(str(e))
        return

    expected_columns = {'k', 'rel_diff'}
    if not expected_columns.issubset(df.columns):
        print("Error: CSV file does not contain expected columns ('k', 'rel_diff'). Found: " + str(list(df.columns)))
        return
    print("Expected columns ('k', 'rel_diff') present: Yes")

    expected_rows = 200
    if df.shape[0] != expected_rows:
        print("Error: CSV file does not have the expected number of rows. Found: " + str(df.shape[0]) + ", Expected: " + str(expected_rows))
        return
    print("Number of rows: " + str(df.shape[0]) + " (Expected: " + str(expected_rows) + ")")

    if df['k'].isnull().any():
        print("Error: 'k' column contains NaN values.")
        return
    print("NaNs in 'k' column: No")

    nan_in_rel_diff_count = df['rel_diff'].isnull().sum()
    print("NaNs in 'rel_diff' column: " + str(nan_in_rel_diff_count))
    
    print("CSV check passed.")
    print("-------------------------------\n")

    # Extract data
    k_values = df['k'].values  # units: h/Mpc
    rel_diff_values = df['rel_diff'].values  # dimensionless

    # --- Statistical Analysis ---
    print("--- Statistical Summary of Relative Difference ---")
    
    avg_rel_diff = np.nanmean(rel_diff_values)
    print("Average relative difference: " + str(avg_rel_diff))

    if np.all(np.isnan(rel_diff_values)):
        print("All relative difference values are NaN. Cannot determine min/max.")
    else:
        max_rel_diff = np.nanmax(rel_diff_values)
        k_at_max_rel_diff = k_values[np.nanargmax(rel_diff_values)]
        print("Maximum relative difference: " + str(max_rel_diff) + " at k = " + str(k_at_max_rel_diff) + " h/Mpc")

        min_rel_diff = np.nanmin(rel_diff_values)
        k_at_min_rel_diff = k_values[np.nanargmin(rel_diff_values)]
        print("Minimum relative difference: " + str(min_rel_diff) + " at k = " + str(k_at_min_rel_diff) + " h/Mpc")
    
    print("------------------------------------------------\n")

    # --- Plotting ---
    print("--- Plotting Relative Difference ---")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, rel_diff_values, marker='.', linestyle='-', markersize=4)
    
    ax.set_xscale('log')
    ax.set_xlabel('k (h/Mpc)')
    ax.set_ylabel('Relative Difference ((P_inv / P_nor) - 1)')
    ax.set_title('Relative Difference in P(k) (Inverted vs Normal Hierarchy)')
    ax.grid(True, which="both", ls="--", alpha=0.7)
    
    fig.tight_layout()
    
    timestamp = time.strftime("%Y%m%d%H%M%S")
    plot_filename = "relative_difference_power_spectrum_1_" + timestamp + ".png"
    plot_filepath = os.path.join(output_dir, plot_filename)
    
    try:
        plt.savefig(plot_filepath, dpi=300)
        print("Plot saved to: " + plot_filepath)
        print("Plot description: The plot shows the relative difference in the matter power spectrum between inverted and normal neutrino hierarchies as a function of wavenumber k (log scale for k-axis).")
    except Exception as e:
        print("Error saving plot:")
        print(str(e))
    
    plt.close(fig)  # Close the figure to free memory
    print("----------------------------------\n")


if __name__ == '__main__':
    analyze_power_spectrum_results()