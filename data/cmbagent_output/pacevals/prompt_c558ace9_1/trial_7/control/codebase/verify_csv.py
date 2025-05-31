# filename: codebase/verify_csv.py
import pandas as pd
import numpy as np
import os

def verify_csv_output():
    """
    Verifies the output CSV file 'result.csv'.
    Checks for:
    - File existence.
    - Correct number of rows (200).
    - Correct column names ('k', 'rel_diff').
    - Numeric data types for columns.
    - Plausible range for 'k' values (approx. 1e-4 to 2.0 h/Mpc).
    - Sanity check on the magnitude of 'rel_diff'.
    """
    database_path = "data"
    filename = "result.csv"
    filepath = os.path.join(database_path, filename)

    kh_min_expected = 1e-4  # h/Mpc
    kh_max_expected = 2.0    # h/Mpc
    n_kpoints_expected = 200
    
    print("Starting verification of: " + filepath)

    # Check 1: File existence
    if not os.path.exists(filepath):
        print("Error: File not found: " + filepath)
        return

    print("Check 1: File exists - PASSED")

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print("Error: Could not read CSV file: " + str(e))
        return

    # Check 2: Number of rows
    if len(df) == n_kpoints_expected:
        print("Check 2: Number of rows (" + str(len(df)) + ") is correct - PASSED")
    else:
        print("Check 2: Number of rows is " + str(len(df)) + ", expected " + str(n_kpoints_expected) + " - FAILED")
        return  # Stop further checks if row count is wrong

    # Check 3: Column names
    expected_columns = ['k', 'rel_diff']
    if list(df.columns) == expected_columns:
        print("Check 3: Column names " + str(list(df.columns)) + " are correct - PASSED")
    else:
        print("Check 3: Column names are " + str(list(df.columns)) + ", expected " + str(expected_columns) + " - FAILED")
        return

    # Check 4: Data types
    k_is_numeric = pd.api.types.is_numeric_dtype(df['k'])
    rel_diff_is_numeric = pd.api.types.is_numeric_dtype(df['rel_diff'])

    if k_is_numeric and rel_diff_is_numeric:
        print("Check 4: Data types for 'k' and 'rel_diff' are numeric - PASSED")
    else:
        if not k_is_numeric:
            print("Check 4: Data type for 'k' is NOT numeric (" + str(df['k'].dtype) + ") - FAILED")
        if not rel_diff_is_numeric:
            print("Check 4: Data type for 'rel_diff' is NOT numeric (" + str(df['rel_diff'].dtype) + ") - FAILED")
        return

    # Check 5: k-value range
    # CAMB k-values are linearly spaced.
    min_k_actual = df['k'].min()
    max_k_actual = df['k'].max()
    tolerance = 1e-9  # Tolerance for float comparison

    k_min_check = abs(min_k_actual - kh_min_expected) < tolerance
    k_max_check = abs(max_k_actual - kh_max_expected) < tolerance
    
    if k_min_check and k_max_check:
        print("Check 5: k-value range (min: " + str(round(min_k_actual,6)) + ", max: " + str(round(max_k_actual,2)) + ") is correct - PASSED")
    else:
        if not k_min_check:
            print("Check 5: Minimum k value is " + str(min_k_actual) + ", expected " + str(kh_min_expected) + " - FAILED")
        if not k_max_check:
            print("Check 5: Maximum k value is " + str(max_k_actual) + ", expected " + str(kh_max_expected) + " - FAILED")
        # Don't return here, other checks might still be useful

    # Check 6: Sanity check on relative difference magnitude
    # For mnu_sum = 0.11 eV, differences are typically small, < 1-2%
    # Let's check if values are within +/- 5% as a broad sanity check.
    min_rel_diff = df['rel_diff'].min()
    max_rel_diff = df['rel_diff'].max()
    
    # Expected range for rel_diff is typically small, e.g., within [-0.02, 0.02]
    # A slightly wider sanity check range:
    sane_min_rd = -0.05 
    sane_max_rd = 0.05

    if sane_min_rd <= min_rel_diff <= sane_max_rd and sane_min_rd <= max_rel_diff <= sane_max_rd:
        print("Check 6: Relative difference values (min: " + str(round(min_rel_diff, 5)) + ", max: " + str(round(max_rel_diff, 5)) + ") are within plausible range [" + str(sane_min_rd) + ", " + str(sane_max_rd) + "] - PASSED")
    else:
        print("Check 6: Relative difference values (min: " + str(min_rel_diff) + ", max: " + str(max_rel_diff) + ") are outside plausible range [" + str(sane_min_rd) + ", " + str(sane_max_rd) + "] - FAILED (This might indicate an issue or unexpected physical result)")

    print("\nVerification Summary:")
    if os.path.exists(filepath) and \
       len(df) == n_kpoints_expected and \
       list(df.columns) == expected_columns and \
       k_is_numeric and rel_diff_is_numeric and \
       k_min_check and k_max_check and \
       (sane_min_rd <= min_rel_diff <= sane_max_rd and sane_min_rd <= max_rel_diff <= sane_max_rd) :
        print("All checks passed. The file " + filename + " appears to be correctly formatted and contains plausible data.")
        print("k values are in h/Mpc.")
        print("rel_diff is dimensionless: (P(k)_inverted / P(k)_normal - 1).")
    else:
        print("One or more checks failed. Please review the output above.")

    print("\nFirst 5 rows of the CSV file:")
    print(df.head())
    print("\nLast 5 rows of the CSV file:")
    print(df.tail())


if __name__ == '__main__':
    verify_csv_output()