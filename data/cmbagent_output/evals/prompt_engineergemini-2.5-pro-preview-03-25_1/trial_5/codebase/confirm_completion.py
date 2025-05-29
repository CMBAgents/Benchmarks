# filename: codebase/confirm_completion.py
import os
import pandas as pd

def confirm_completion():
    r""" 
    Confirms the completion of the CMB power spectrum calculation and saving task.

    This function checks if the expected output file 'data/result.csv' exists,
    is readable, and contains data. It prints messages to the console
    indicating the status of this verification.
    """
    file_path = os.path.join('data', 'result.csv')

    print("Attempting to confirm completion of CMB power spectrum calculation...")

    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            print("Confirmation: Output file 'data/result.csv' exists and is readable.")
            
            if not df.empty:
                print("Number of rows in 'data/result.csv': " + str(len(df)))
                print("First row of 'data/result.csv':")
                # Ensure 'l' is printed as int if it is
                if 'l' in df.columns and pd.api.types.is_numeric_dtype(df['l']):
                     df_display_head = df.head(1).copy()
                     df_display_head['l'] = df_display_head['l'].astype(int)
                     print(df_display_head.to_string(index=False))
                else:
                     print(df.head(1).to_string(index=False))
                
                print("Last row of 'data/result.csv':")
                if 'l' in df.columns and pd.api.types.is_numeric_dtype(df['l']):
                     df_display_tail = df.tail(1).copy()
                     df_display_tail['l'] = df_display_tail['l'].astype(int)
                     print(df_display_tail.to_string(index=False))
                else:
                    print(df.tail(1).to_string(index=False))

                # Check if expected columns are present
                expected_columns = ['l', 'TT']
                if all(col in df.columns for col in expected_columns):
                    print("Expected columns 'l' and 'TT' are present in the CSV file.")
                else:
                    print("Warning: Expected columns 'l' and 'TT' might be missing. Found columns: " + str(list(df.columns)))
                
                print("The main task of calculating and saving the CMB power spectrum appears to be successfully completed.")
            else:
                print("Warning: 'data/result.csv' is empty.")
                print("The main task of calculating and saving the CMB power spectrum was initiated, but the output file contains no data.")
        except pd.errors.EmptyDataError:
            print("Warning: 'data/result.csv' is empty and could not be parsed by pandas.")
            print("The main task of calculating and saving the CMB power spectrum was initiated, but the output file is empty.")
        except Exception as e:
            print("Error reading or processing 'data/result.csv' for confirmation: " + str(e))
            print("The main task of calculating and saving the CMB power spectrum was initiated, but confirmation of output failed.")
    else:
        print("Warning: Output file 'data/result.csv' not found.")
        print("The previous step (CMB calculation and saving) might not have completed successfully or the file was saved elsewhere.")

if __name__ == '__main__':
    # Set pandas display options for better console output of DataFrame snippets
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.width', 120) 
    pd.set_option('display.max_colwidth', 100)
    pd.set_option('display.colheader_justify', 'left')
    
    confirm_completion()