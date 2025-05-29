# filename: codebase/analyze_cmb.py
import pandas as pd


def main():
    # Define the path to the CSV file relative to the codebase directory
    data_path = '../data/result.csv'
    
    # Read the CSV data
    df = pd.read_csv(data_path)
    
    # Print the first few rows for verification
    print('First few rows of the result:')
    print(df.head())
    
    # Print the last few rows for verification
    print('\nLast few rows of the result:')
    print(df.tail())


if __name__ == '__main__':
    main()