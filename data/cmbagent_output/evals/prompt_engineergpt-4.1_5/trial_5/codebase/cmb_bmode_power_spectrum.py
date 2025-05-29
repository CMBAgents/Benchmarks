# filename: codebase/cmb_bmode_power_spectrum.py
import pandas as pd

# Read the CSV file with the CMB B-mode polarization power spectrum data
# Columns: 'l' for multipole moments and 'BB' for the corresponding B-mode power spectrum values in microK^2

data_file = 'data/result.csv'
df = pd.read_csv(data_file)

# Print a summary of the results
print('First 10 rows:')
print(df.head(10))

print('Last 10 rows:')
print(df.tail(10))
