# filename: codebase/plot_matter_power_spectrum.py
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

# Define the input CSV file and output plot directory
output_dir = "data"
csv_filename = os.path.join(output_dir, "result.csv")
plot_filename_base = "relative_difference_plot"

def plot_relative_difference():
    """
    Reads the relative difference data from result.csv and plots it.
    Saves the plot to a PNG file and prints an interpretation.
    """
    try:
        # Read the CSV file
        results_df = pd.read_csv(csv_filename)
        print("Successfully loaded data from " + csv_filename)
    except FileNotFoundError:
        print("Error: The file " + csv_filename + " was not found.")
        print("Please ensure that the previous step (Step 2) was executed successfully and generated this file.")
        return
    except Exception as e:
        print("An error occurred while reading " + csv_filename + ": " + str(e))
        return

    if results_df.empty:
        print("The DataFrame loaded from " + csv_filename + " is empty. Cannot generate plot.")
        return
        
    if 'k' not in results_df.columns or 'rel_diff' not in results_df.columns:
        print("Error: The CSV file " + csv_filename + " does not contain the required columns 'k' and 'rel_diff'.")
        print("Columns found: " + str(results_df.columns.tolist()))
        return

    # Ensure matplotlib does not use LaTeX rendering
    plt.rcParams['text.usetex'] = False

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(results_df['k'], results_df['rel_diff'], label='(P_inv / P_nor) - 1')

    # Set plot labels and title
    ax.set_xlabel("Wavenumber, k (h/Mpc)")
    ax.set_ylabel("Relative Difference ((P_inv / P_nor) - 1)")
    ax.set_title("Relative Difference in Matter Power Spectrum (Inverted vs. Normal Hierarchy)")

    # Use a log scale for the x-axis as k spans orders of magnitude
    ax.set_xscale('log')

    # Add a grid for better readability
    ax.grid(True, which="both", ls="--", alpha=0.7)
    
    ax.legend()

    # Ensure everything fits without overlapping
    fig.tight_layout()

    # Save the plot
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plot_number = 1 # Assuming this is the first plot of this kind
    plot_filename = os.path.join(output_dir, plot_filename_base + "_" + str(plot_number) + "_" + timestamp + ".png")

    try:
        plt.savefig(plot_filename, dpi=300)
        print("Plot saved as: " + plot_filename)
        print("Description of the plot:")
        print("- The plot shows the relative difference in the linear matter power spectrum between the inverted and normal neutrino hierarchies as a function of wavenumber k.")
        print("- The x-axis (wavenumber k) is on a logarithmic scale.")
        print("- The y-axis represents (P(k)_inverted / P(k)_normal - 1).")
        
        # Brief interpretation based on expected physics
        print("\nBrief Interpretation of the Results:")
        print("Neutrinos, being massive particles, affect the growth of cosmic structures.")
        print("Their free-streaming nature tends to suppress power on scales smaller than their free-streaming length (larger k values).")
        print("The specific distribution of the total neutrino mass among the three eigenstates (i.e., the hierarchy - normal vs. inverted) leads to subtle differences in this suppression.")
        print("For a fixed total neutrino mass (0.11 eV in this case):")
        print("  - In the normal hierarchy, one neutrino is significantly heavier than the other two (m1 < m2 << m3).")
        print("  - In the inverted hierarchy, two neutrinos are nearly degenerate and heavier, while one is much lighter (m3 << m1 ~ m2).")
        print("This difference in mass distribution can cause a k-dependent variation in the matter power spectrum.")
        print("The plot visualizes this fractional difference. Typically, the differences are at the sub-percent to percent level.")
        print("Observe the plot for specific features: ")
        print("  - At very small k (large scales), the difference is usually negligible as both hierarchies behave similarly to a small correction to CDM.")
        print("  - As k increases (smaller scales), the differences due to neutrino free-streaming become more apparent.")
        print("  - The sign and magnitude of the relative difference depend on the interplay of the individual neutrino masses and their contribution to the total energy density and suppression effects.")
        
        # Add some quantitative observations if data allows (e.g. max difference)
        if not results_df['rel_diff'].isnull().all():
            max_abs_rel_diff = results_df['rel_diff'].abs().max()
            k_at_max_abs_rel_diff = results_df.loc[results_df['rel_diff'].abs().idxmax(), 'k']
            print("The maximum absolute relative difference observed is approximately " + "%.2e" % max_abs_rel_diff + " at k approx " + "%.2e" % k_at_max_abs_rel_diff + " h/Mpc.")
        else:
            print("Relative difference data contains NaNs or is empty, cannot provide quantitative summary.")


    except Exception as e:
        print("Error saving plot " + plot_filename + ": " + str(e))
    
    # Close the plot to free memory
    plt.close(fig)


if __name__ == "__main__":
    # Create output directory if it doesn't exist (though previous step should have)
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print("Created directory: " + output_dir)
        except OSError as e:
            print("Error creating directory " + output_dir + ": " + str(e))
            # If directory creation fails, exit or handle appropriately
            # For this plotting script, we might still proceed if csv_filename is absolute path
            # but it's better to ensure the dir exists for saving the plot.
            
    plot_relative_difference()
    print("\nPlotting script finished.")