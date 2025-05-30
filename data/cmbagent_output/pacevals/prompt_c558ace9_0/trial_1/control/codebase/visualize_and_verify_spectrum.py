# filename: codebase/visualize_and_verify_spectrum.py
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import numpy as np


def visualize_and_verify_spectrum(database_path="data"):
    """
    Verifies the content of the CMB power spectrum CSV file (result.csv)
    and generates a plot of the spectrum.

    The CSV file is expected to have two columns: 'l' (multipole moment)
    and 'TT' (raw temperature power spectrum in muK^2).

    Args:
        database_path (str): The path to the directory containing 'result.csv'
                             and where the plot will be saved. Defaults to "data".
    """
    file_path = os.path.join(database_path, "result.csv")

    # --- CSV Verification ---
    print("Starting verification of: " + file_path)
    if not os.path.exists(file_path):
        print("Error: CSV file not found at " + file_path)
        print("Please ensure the CAMB calculation has been run and result.csv is generated.")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print("Error reading CSV file: " + file_path)
        print(str(e))
        return

    # Check headers
    expected_headers = ['l', 'TT']
    if not df.columns.tolist() == expected_headers:
        print("Error: CSV headers are incorrect.")
        print("Expected: " + str(expected_headers))
        print("Found: " + str(df.columns.tolist()))
        return
    print("CSV headers ('l', 'TT') are correct.")

    # Verify 'l' column
    if not pd.api.types.is_integer_dtype(df['l']):
        print("Error: 'l' column is not of integer type.")
        return
    
    expected_l_values = np.arange(2, 3001)
    if not np.array_equal(df['l'].values, expected_l_values):
        print("Error: 'l' column values are not a sequential range from 2 to 3000.")
        # Further diagnostics (optional, can be verbose)
        if df['l'].min() != 2:
            print("Min 'l' is " + str(df['l'].min()) + ", expected 2.")
        if df['l'].max() != 3000:
            print("Max 'l' is " + str(df['l'].max()) + ", expected 3000.")
        if len(df['l']) != len(expected_l_values):
            print("Length of 'l' column is " + str(len(df['l'])) + ", expected " + str(len(expected_l_values)) + ".")
        if not df['l'].is_monotonic_increasing:
            print("'l' column is not monotonically increasing.")
        if df['l'].nunique() != len(df['l']):
            print("'l' column contains duplicate values.")
        return
    print("'l' column verified (integer, sequential from 2 to 3000).")

    # Verify 'TT' column
    if not pd.api.types.is_numeric_dtype(df['TT']):
        print("Error: 'TT' column is not of numeric type.")
        return
    if not (df['TT'] >= 0).all():
        print("Warning: 'TT' column contains negative values. Power spectrum values should be non-negative.")
        # Depending on severity, one might choose to return here or just warn.
        # For now, we'll just warn and proceed with plotting.
    print("'TT' column verified (numeric, non-negative values checked).")

    print("CSV file verification successful.")

    # --- Plotting ---
    print("Generating CMB power spectrum plot...")
    try:
        plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering

        fig, ax = plt.subplots(figsize=(10, 6))

        l_values = df['l']
        tt_powers = df['TT']  # Units: muK^2

        ax.plot(l_values, tt_powers)

        ax.set_xlabel("Multipole moment $l$")
        ax.set_ylabel("TT Power Spectrum $C_l^{TT}$ ($\\mu K^2$)")
        ax.set_title("CMB Raw Temperature Power Spectrum")

        ax.set_yscale('log')  # Power spectrum often plotted on log scale for y-axis
        ax.grid(True, which="both", ls="-", alpha=0.7)
        
        # Ensure x-axis starts near 0 or 2, depending on preference for l=0,1 if they were present
        # Since our data starts at l=2, this should be fine.
        ax.set_xlim(left=0, right=l_values.max() + 100)  # Extend x-axis slightly beyond max l

        plt.tight_layout()

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        plot_filename = "cmb_power_spectrum_plot_1_" + timestamp_str + ".png"
        plot_filepath = os.path.join(database_path, plot_filename)

        plt.savefig(plot_filepath, dpi=300)
        plt.close(fig)  # Close the figure to free memory

        print("Plot saved to: " + plot_filepath)
        print("Plot description: CMB Raw Temperature Power Spectrum ($C_l^{TT}$ vs. $l$). " +
              "Y-axis ($C_l^{TT}$) is on a log scale. Data sourced from " + file_path + ".")

    except Exception as e:
        print("An error occurred during plotting:")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # Define the base path for data files and plots.
    # As per instructions, files are saved in "data/".
    database_path_main = "data"

    # Ensure the data directory exists (it should if result.csv was created by a previous step)
    if not os.path.exists(database_path_main):
        os.makedirs(database_path_main)
        print("Created directory: " + database_path_main)
    
    try:
        visualize_and_verify_spectrum(database_path=database_path_main)
    except FileNotFoundError:
        # This specific case is handled inside visualize_and_verify_spectrum,
        # but good to have a general catch here too.
        print("Error: The result.csv file was not found. Ensure it is generated in the '" + database_path_main + "' directory.")
    except Exception as e:
        print("An unexpected error occurred:")
        import traceback
        traceback.print_exc()
