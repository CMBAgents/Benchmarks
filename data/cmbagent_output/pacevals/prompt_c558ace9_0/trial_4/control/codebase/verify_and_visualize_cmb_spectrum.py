# filename: codebase/verify_and_visualize_cmb_spectrum.py
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import numpy as np

# Disable LaTeX rendering for matplotlib
plt.rcParams['text.usetex'] = False

def verify_and_visualize_cmb_spectrum():
    """
    Verifies the CMB power spectrum data from 'result.csv' and creates a plot.

    The script performs the following actions:
    1. Loads data from 'data/result.csv'.
    2. Verifies column names, data types, range of multipole moments (l),
       and checks for missing values.
    3. Prints a summary of verification checks and key values from the spectrum
       (e.g., Sachs-Wolfe plateau, first acoustic peak) to help confirm units (muK^2).
    4. Generates a plot of C_l^TT vs. l (with y-axis on a log scale).
    5. Saves the plot to 'data/cmb_power_spectrum_1_<timestamp>.png'.
    6. Prints a description of the plot.
    """
    print("Starting verification and visualization of CMB power spectrum...")

    database_path = "data"
    csv_filename = "result.csv"
    csv_filepath = os.path.join(database_path, csv_filename)

    # Check if the CSV file exists
    if not os.path.exists(csv_filepath):
        print("Error: CSV file not found at " + csv_filepath)
        print("Please ensure 'result.csv' was generated successfully in the previous step.")
        return

    print("Loading data from " + csv_filepath + "...")
    try:
        # Read the CSV, ensuring 'l' is integer and 'TT' is float
        # The header was 'l,TT' and comments=''
        df = pd.read_csv(csv_filepath, header=0, names=['l', 'TT'], 
                         dtype={'l': int, 'TT': float}, comment=None)
        print("Data loaded successfully.")
    except Exception as e:
        print("Error loading CSV file: " + str(e))
        return

    # --- Verification Checks ---
    print("\n--- Data Verification ---")
    valid_data = True

    # 1. Check column names
    expected_columns = ['l', 'TT']
    if list(df.columns) != expected_columns:
        print("Error: CSV columns are not as expected. Found: " + str(list(df.columns)) +
              ", Expected: " + str(expected_columns))
        valid_data = False
    else:
        print("CSV columns ('l', 'TT') are correct.")

    # 2. Check data types (already enforced by dtype in read_csv, but good to confirm)
    if 'l' in df.columns and df['l'].dtype == np.int64 or df['l'].dtype == np.int32 : # pandas uses int64 by default
        print("Column 'l' has integer data type: Correct.")
    elif 'l' in df.columns :
        print("Warning: Column 'l' data type is " + str(df['l'].dtype) + ", expected integer.")
        # Attempt to convert, or flag as error depending on strictness
        try:
            df['l'] = pd.to_numeric(df['l']).astype(int)
            print("Attempted conversion of 'l' to integer.")
        except ValueError:
            print("Error: Column 'l' could not be converted to integer.")
            valid_data = False


    if 'TT' in df.columns and df['TT'].dtype == np.float64:
        print("Column 'TT' has float data type: Correct.")
    elif 'TT' in df.columns:
        print("Warning: Column 'TT' data type is " + str(df['TT'].dtype) + ", expected float.")
        try:
            df['TT'] = pd.to_numeric(df['TT']).astype(float)
            print("Attempted conversion of 'TT' to float.")
        except ValueError:
            print("Error: Column 'TT' could not be converted to float.")
            valid_data = False
            
    if not valid_data:
        print("Halting due to critical data type issues.")
        return


    # 3. Check range of 'l'
    expected_l_min = 2
    expected_l_max = 3000
    actual_l_min = df['l'].min()
    actual_l_max = df['l'].max()

    if actual_l_min == expected_l_min and actual_l_max == expected_l_max:
        print("Multipole moment 'l' ranges from " + str(actual_l_min) + " to " + str(actual_l_max) + ": Correct.")
    else:
        print("Error: Multipole moment 'l' range is incorrect. Found: " +
              str(actual_l_min) + "-" + str(actual_l_max) +
              ", Expected: " + str(expected_l_min) + "-" + str(expected_l_max))
        valid_data = False

    # 4. Check number of data points
    expected_rows = expected_l_max - expected_l_min + 1
    actual_rows = len(df)
    if actual_rows == expected_rows:
        print("Number of data points is " + str(actual_rows) + ": Correct.")
    else:
        print("Error: Number of data points is incorrect. Found: " + str(actual_rows) +
              ", Expected: " + str(expected_rows))
        valid_data = False
        
    # 5. Check for NaN values in 'TT'
    nan_tt_count = df['TT'].isnull().sum()
    if nan_tt_count == 0:
        print("No missing values (NaN) in 'TT' column: Correct.")
    else:
        print("Error: Found " + str(nan_tt_count) + " missing values (NaN) in 'TT' column.")
        valid_data = False

    if not valid_data:
        print("\nOne or more verification checks failed. Please review the CSV file.")
        # We can still try to plot if some data is there
    else:
        print("\nAll verification checks passed. Data appears to be correctly formatted.")

    # --- Magnitude and Unit Confirmation (using C_l^TT in muK^2) ---
    print("\n--- Magnitude Confirmation (Units: muK^2) ---")
    if 'l' in df.columns and 'TT' in df.columns and not df.empty:
        print("First few C_l^TT values (Sachs-Wolfe Plateau):")
        for i in range(min(5, len(df))):
            print("l=" + str(df['l'].iloc[i]) + ", C_l^TT = " + str(df['TT'].iloc[i]) + " muK^2")

        # Find the first acoustic peak (approximate)
        if not df['TT'].empty:
            peak_tt_value = df['TT'].max()
            peak_l_value = df.loc[df['TT'].idxmax(), 'l']
            print("\nApproximate First Acoustic Peak:")
            print("Max C_l^TT = " + str(peak_tt_value) + " muK^2 at l = " + str(peak_l_value))
            # Typical C_l^TT at first peak is ~4000-6000 muK^2 around l~220.
            # The values from CAMB are directly C_l in muK^2.
            if 200 <= peak_l_value <= 250 and 3000 < peak_tt_value < 7000:
                 print("Peak magnitude and position are within typical ranges for C_l^TT in muK^2.")
            else:
                 print("Warning: Peak magnitude or position may differ from typical expectations. " +
                       "Expected l ~ 220, C_l^TT ~ 4000-6000 muK^2. Check parameters if concerned.")
        else:
            print("TT column is empty, cannot determine peak.")

    else:
        print("Cannot perform magnitude check as data is missing or columns are incorrect.")


    # --- Plotting ---
    print("\n--- Plotting CMB Power Spectrum ---")
    if 'l' in df.columns and 'TT' in df.columns and not df.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(df['l'], df['TT'])
        plt.yscale('log')  # C_l^TT spans orders of magnitude
        plt.xscale('linear') # l up to 3000, linear is fine. Could use 'log' for l > ~50.
        
        plt.xlabel("Multipole moment $l$")
        plt.ylabel("$C_l^{TT}$ [$\mu K^2$]") # Using brackets for units
        plt.title("CMB Temperature Power Spectrum ($C_l^{TT}$)")
        plt.grid(True, which="both", ls="-", alpha=0.5) # Add grid lines
        
        plt.tight_layout() # Adjust plot to prevent labels from overlapping

        # Save the plot
        timestamp = time.strftime("%Y%m%d%H%M%S")
        plot_filename = "cmb_power_spectrum_1_" + timestamp + ".png"
        plot_filepath = os.path.join(database_path, plot_filename)
        
        try:
            plt.savefig(plot_filepath, dpi=300)
            print("Plot saved successfully to: " + plot_filepath)
            print("Plot Description: This plot shows the CMB temperature power spectrum (C_l^TT) " +
                  "as a function of the multipole moment (l). The y-axis (C_l^TT, in muK^2) " +
                  "is on a logarithmic scale, while the x-axis (l) is linear. " +
                  "It displays the Sachs-Wolfe plateau, acoustic peaks, and damping tail, " +
                  "characteristic of CMB anisotropy.")
        except Exception as e:
            print("Error saving plot: " + str(e))
        
        # Do not use plt.show()
        plt.close() # Close the plot figure to free memory

    else:
        print("Cannot generate plot as data is missing or columns 'l' and 'TT' are not available.")

    print("\nVerification and visualization process completed.")


if __name__ == '__main__':
    # Create data directory if it doesn't exist, for local testing
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Created directory: data (for local testing)")
        # Create a dummy result.csv for testing if previous step didn't run
        # This part would typically not be in the final script for the agent
        # but is useful for standalone testing of this step.
        # For agent execution, result.csv is expected to exist.
        if not os.path.exists(os.path.join("data", "result.csv")):
            print("Creating a dummy result.csv for testing purposes...")
            dummy_l = np.arange(2, 3001)
            # Simplified C_l^TT: SW plateau, one peak, exponential decay
            dummy_tt = 1000 * np.exp(- (dummy_l - 220)**2 / (2 * 80**2)) + \
                       5000 * np.exp(- (dummy_l - 220)**2 / (2 * 30**2)) + \
                       100 * np.exp(-dummy_l / 500.0) + \
                       np.random.rand(len(dummy_l)) * 10
            dummy_tt[:50] = 1000 + np.random.rand(50)*100 # SW plateau
            dummy_data = pd.DataFrame({'l': dummy_l, 'TT': dummy_tt})
            dummy_data.to_csv(os.path.join("data", "result.csv"), index=False, header=True, float_format='%.18e')


    verify_and_visualize_cmb_spectrum()
