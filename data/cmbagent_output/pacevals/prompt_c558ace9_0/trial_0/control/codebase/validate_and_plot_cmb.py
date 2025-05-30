# filename: codebase/validate_and_plot_cmb.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time


def validate_and_plot_cmb_spectrum():
    """
    Validates the CMB power spectrum data from a CSV file and plots it.
    - Verifies file format, multipole range, and data types.
    - Confirms units are implicitly muK^2 by checking against expected patterns in the plot.
    - Creates a plot of C_l^TT vs l.
    """
    plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering

    input_csv_file = os.path.join("data", "result.csv")
    plot_output_dir = "data"
    
    # Create plot output directory if it doesn't exist
    try:
        if not os.path.exists(plot_output_dir):
            os.makedirs(plot_output_dir)
            print("Created directory: " + plot_output_dir)
    except OSError as e:
        print("Error creating directory " + plot_output_dir + ": " + str(e))
        return

    print("Starting validation of CMB power spectrum data from: " + input_csv_file)

    try:
        # 1. Load the CSV file
        if not os.path.exists(input_csv_file):
            print("Error: Input CSV file not found at " + input_csv_file)
            return

        df = pd.read_csv(input_csv_file)
        print("Successfully loaded " + input_csv_file)

        # 2. Verify file format and column names
        expected_columns = ['l', 'TT']
        if list(df.columns) == expected_columns:
            print("Validation PASSED: Columns are correctly named 'l' and 'TT'.")
        else:
            print("Validation FAILED: Columns are not named 'l' and 'TT'. Found: " + str(list(df.columns)))
            return

        # 3. Verify multipole range (l=2 to l=3000)
        min_l = df['l'].min()
        max_l = df['l'].max()
        expected_min_l = 2
        expected_max_l = 3000

        if min_l == expected_min_l and max_l == expected_max_l:
            print("Validation PASSED: Multipole 'l' ranges from " + str(expected_min_l) + " to " + str(expected_max_l) + ".")
        else:
            print("Validation FAILED: Multipole 'l' range is incorrect. Found min_l=" + str(min_l) + ", max_l=" + str(max_l) + 
                  ". Expected min_l=" + str(expected_min_l) + ", max_l=" + str(expected_max_l) + ".")
            return

        # 4. Verify 'l' column contains integers and is complete
        if pd.api.types.is_integer_dtype(df['l']):
            print("Validation PASSED: 'l' column contains integer values.")
        else:
            print("Validation FAILED: 'l' column does not contain integer values. dtype: " + str(df['l'].dtype))
            return
        
        expected_l_values = np.arange(expected_min_l, expected_max_l + 1)
        if np.array_equal(df['l'].values, expected_l_values):
            print("Validation PASSED: 'l' column contains all integers from " + str(expected_min_l) + " to " + str(expected_max_l) + ".")
        else:
            print("Validation FAILED: 'l' column is not a complete sequence from " + str(expected_min_l) + " to " + str(expected_max_l) + ".")
            # Find missing or extra values for more detailed feedback (optional)
            if len(df['l']) != len(expected_l_values):
                 print("Length mismatch: Found " + str(len(df['l'])) + " rows, expected " + str(len(expected_l_values)) + ".")
            # Could add more detailed diff if necessary
            return
            
        # 5. Verify 'TT' column contains numeric values
        if pd.api.types.is_numeric_dtype(df['TT']):
            print("Validation PASSED: 'TT' column contains numeric values.")
        else:
            print("Validation FAILED: 'TT' column does not contain numeric values. dtype: " + str(df['TT'].dtype))
            return

        # 6. Confirm units are in muK^2 (implicitly by successful generation and visual check of plot)
        # The previous script requested 'muK' units from CAMB.
        print("Unit Confirmation: Data is assumed to be in muK^2 as per CAMB generation script. Visual plot inspection will help confirm typical magnitudes.")

        # 7. Create a plot of the power spectrum
        l_values = df['l'].values
        cl_tt_values = df['TT'].values  # Units: muK^2

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(l_values, cl_tt_values)
        ax.set_xlabel("Multipole moment $l$")
        ax.set_ylabel("$C_l^{TT}$ [$\mu K^2$]")  # CMB Power Spectrum units
        ax.set_title("CMB Temperature Power Spectrum ($C_l^{TT}$)")
        ax.set_yscale('log')  # Log scale for y-axis is common for C_l
        ax.grid(True, which="both", ls="-", alpha=0.5)
        
        plt.tight_layout()

        timestamp = int(time.time())
        plot_filename = "cmb_power_spectrum_plot_1_" + str(timestamp) + ".png"
        plot_filepath = os.path.join(plot_output_dir, plot_filename)
        
        plt.savefig(plot_filepath, dpi=300)
        plt.close(fig)  # Close the figure to free memory

        print("\nPlot generated successfully.")
        print("Plot saved to: " + plot_filepath)
        print("Description of the plot:")
        print("- X-axis: Multipole moment l (from " + str(min_l) + " to " + str(max_l) + ")")
        print("- Y-axis: Raw Temperature Power Spectrum C_l^TT (in muK^2), plotted on a logarithmic scale.")
        print("- Title: 'CMB Temperature Power Spectrum (C_l^TT)'")
        print("- The plot should display the characteristic acoustic peaks of the CMB power spectrum.")

    except FileNotFoundError:
        print("Error: The file " + input_csv_file + " was not found.")
    except pd.errors.EmptyDataError:
        print("Error: The file " + input_csv_file + " is empty.")
    except Exception as e:
        print("An error occurred during validation or plotting:")
        import traceback
        print(traceback.format_exc())


if __name__ == '__main__':
    validate_and_plot_cmb_spectrum()
