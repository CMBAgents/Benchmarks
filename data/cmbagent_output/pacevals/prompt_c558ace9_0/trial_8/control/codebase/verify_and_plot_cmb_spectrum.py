# filename: codebase/verify_and_plot_cmb_spectrum.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime

# Ensure LaTeX rendering is disabled for matplotlib
matplotlib.rcParams['text.usetex'] = False

def verify_and_plot_cmb_spectrum():
    """
    Verifies the 'result.csv' file containing CMB power spectrum data
    and generates a plot of D_l^TT vs l.

    Verification steps:
    1. Checks if 'data/result.csv' exists.
    2. Validates column names ('l', 'TT').
    3. Checks if 'l' ranges from 2 to 3000.
    4. Checks if the number of data points is 2999.
    5. Checks data types: 'l' (integer), 'TT' (numeric).
    6. Checks if C_l^TT values are non-negative.
    7. Performs a basic sanity check on the peak value of D_l^TT.

    If verification is successful, it plots D_l^TT = l(l+1)C_l^TT/(2*pi) vs l
    and saves it to 'data/cmb_power_spectrum_plot_1_<timestamp>.png'.
    """
    print("Starting verification and plotting of CMB power spectrum...")
    database_path = "data"
    csv_filename = os.path.join(database_path, "result.csv")

    all_checks_passed = True

    try:
        # Check 1: File existence
        if not os.path.exists(csv_filename):
            print("Error: " + csv_filename + " not found.")
            all_checks_passed = False
        else:
            print(csv_filename + " found.")
            df = pd.read_csv(csv_filename)

            # Check 2: Column names
            expected_columns = ['l', 'TT']
            if not all(col in df.columns for col in expected_columns):
                print("Error: CSV file does not contain the expected columns 'l' and 'TT'. Found: " + str(list(df.columns)))
                all_checks_passed = False
            else:
                print("CSV columns 'l' and 'TT' found.")

                # Check 3: Multipole range
                min_l = df['l'].min()
                max_l = df['l'].max()
                if min_l != 2 or max_l != 3000:
                    print("Error: Multipole moment 'l' range is incorrect. Expected 2-3000, got " + str(min_l) + "-" + str(max_l) + ".")
                    all_checks_passed = False
                else:
                    print("Multipole moment 'l' range (2-3000) is correct.")

                # Check 4: Number of data points
                expected_rows = 3000 - 2 + 1
                if len(df) != expected_rows:
                    print("Error: Number of data points is incorrect. Expected " + str(expected_rows) + ", got " + str(len(df)) + ".")
                    all_checks_passed = False
                else:
                    print("Number of data points (" + str(len(df)) + ") is correct.")

                # Check 5: Data types
                if not pd.api.types.is_integer_dtype(df['l']):
                    print("Error: Column 'l' is not of integer type. Type found: " + str(df['l'].dtype))
                    all_checks_passed = False
                else:
                    print("Column 'l' is of integer type.")

                if not pd.api.types.is_numeric_dtype(df['TT']): # C_l^TT in muK^2
                    print("Error: Column 'TT' is not of numeric type. Type found: " + str(df['TT'].dtype))
                    all_checks_passed = False
                else:
                    print("Column 'TT' (C_l^TT) is of numeric type.")
                
                # Check 6: Non-negative power
                if (df['TT'] < 0).any():
                    print("Warning: Some C_l^TT values are negative. This might indicate an issue.")
                    # This might not strictly be an error for CAMB output at very low l or numerical precision,
                    # but it's worth noting. For this problem, we expect positive values.
                    # To be stricter, one could set all_checks_passed = False
                else:
                    print("All C_l^TT values are non-negative.")

                # Check 7: Plausible values (Peak Check for D_l^TT)
                # D_l^TT = l(l+1)C_l^TT / (2*pi)
                # C_l^TT is in df['TT']
                l_values = df['l'].values
                cl_TT_values = df['TT'].values # Units: muK^2
                dl_TT_values = l_values * (l_values + 1) * cl_TT_values / (2 * np.pi) # Units: muK^2

                # Find peak around l=200-250
                peak_region = dl_TT_values[(l_values >= 200) & (l_values <= 250)]
                if len(peak_region) > 0:
                    first_acoustic_peak_approx = np.max(peak_region)
                    print("Approximate first acoustic peak D_l^TT (around l=200-250): " + str(first_acoustic_peak_approx) + " muK^2")
                    # Typical values for the first peak are in the thousands.
                    if not (1000 < first_acoustic_peak_approx < 10000):
                        print("Warning: The first acoustic peak value (" + str(first_acoustic_peak_approx) + " muK^2) is outside the typical range (1000-10000 muK^2).")
                        # This is a soft check, could be adjusted.
                    else:
                        print("First acoustic peak value is within the expected rough range.")
                else:
                    print("Warning: Could not find data in the l=200-250 region to check the first acoustic peak.")


            if all_checks_passed:
                print("All verification checks passed.")
                # Proceed to plotting
                
                # Calculate D_l^TT = l(l+1)C_l^TT / (2*pi) for plotting
                # C_l^TT is in df['TT']
                l_plot = df['l'].values
                cl_TT_plot = df['TT'].values # Units: muK^2
                dl_TT_plot = l_plot * (l_plot + 1) * cl_TT_plot / (2 * np.pi) # Units: muK^2

                plt.figure(figsize=(10, 6))
                plt.plot(l_plot, dl_TT_plot)
                plt.xlabel("Multipole moment l")
                plt.ylabel("D_l^TT [l(l+1)C_l^TT / (2π)] (μK^2)")
                plt.title("CMB Temperature Power Spectrum (Unlensed Scalar)")
                plt.grid(True)
                # plt.xscale('log') # Often CMB plots use log scale for l, but linear is fine for verification
                
                plt.tight_layout()

                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                plot_filename = os.path.join(database_path, "cmb_power_spectrum_plot_1_" + timestamp + ".png")
                
                plt.savefig(plot_filename, dpi=300)
                print("Plot saved to: " + plot_filename)
                print("Plot description: CMB Temperature Power Spectrum D_l^TT = l(l+1)C_l^TT/(2pi) vs multipole moment l. " +
                      "D_l^TT is in muK^2. The plot shows the characteristic acoustic peaks.")
            else:
                print("Verification failed. Plot will not be generated.")

    except FileNotFoundError:
        print("Error: " + csv_filename + " not found. Cannot perform verification or plotting.")
    except pd.errors.EmptyDataError:
        print("Error: " + csv_filename + " is empty. Cannot perform verification or plotting.")
    except Exception as e:
        print("An error occurred during verification or plotting:")
        print(str(e))


if __name__ == '__main__':
    verify_and_plot_cmb_spectrum()
    # print("##SCRIPT_COMPLETED_SUCCESSFULLY##") # For automated systems
