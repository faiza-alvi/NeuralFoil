import numpy as np
import pandas as pd
from scipy.special import comb
import polars as pl
import csv


def find_longest_rows(filename):
    """
    This function finds the longest rows in the file
    It returns the indices of the longest rows and the maximum length
    """
    longest_row_indices = []
    max_length = 0

    with open(filename, 'r', newline='') as f_in:
        reader = csv.reader(f_in)
        for idx, row in enumerate(reader, start=1):  # start=1 → human-readable row numbers
            current_length = len(row)
            if current_length > max_length:
                # Found a new max → reset indices
                max_length = current_length
                longest_row_indices = [idx]
            elif current_length == max_length:
                # Found another row with same max length
                longest_row_indices.append(idx)

    return longest_row_indices, max_length

def remove_longest_rows(input_file, output_file):
    """
    Removes any lines that are longer than the other lines and writes an output .csv 
    """
    # First pass: find max length
    max_length = 0
    with open(input_file, 'r', newline='') as f_in:
        reader = csv.reader(f_in)
        for row in reader:
            max_length = max(max_length, len(row))

    # Second pass: write only rows that are NOT longest
    with open(input_file, 'r', newline='') as f_in, \
         open(output_file, 'w', newline='') as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        for row in reader:
            if len(row) != max_length:
                writer.writerow(row)

    return max_length

def clean_csv(input_file, output_file):
    # Read CSV
    df = pd.read_csv(input_file)

    # Select the first 18 columns
    first_18_cols = df.columns[:18]

    # Find rows with nulls in first 18 columns
    bad_rows = df[df[first_18_cols].isnull().any(axis=1)]

    # Print info
    print(f"Rows removed: {len(bad_rows)}")
    if not bad_rows.empty:
        print("Removed row indices:", bad_rows.index.tolist())

    # Drop those rows
    cleaned_df = df.drop(bad_rows.index)

    # Save cleaned CSV
    cleaned_df.to_csv(output_file, index=False)


# -----------------------------------
# Implementation to remove the longest rows and then save the new output datafile after it has been cleaned 

# input_file = r"C:\Users\booki\Documents\BIRD Lab\Airfoil Project\data_xfoil_final_K9.csv"  # Replace with your CSV file name
# output_file = r"C:\Users\booki\Documents\BIRD Lab\Airfoil Project\data_xfoil_final_K9_clean.csv"  # Replace with your CSV file name
# max_len = remove_longest_rows(input_file, output_file)

# print(f"Removed rows with {max_len} columns. New file saved as {output_file}")

# -----------------------------------
# Checks whether there are any more rows that are longer than the rest.
# For generated xfoil data there should be 222 columns.

# file_name = r"C:\Users\booki\Documents\BIRD Lab\Airfoil Project\data_xfoil_final_K9_clean.csv"  # Replace with your CSV file name
# row_indices, length = find_longest_rows(file_name)

# print(f"The longest rows have {length} columns ")
# print(f"The longest rows are at indices: {row_indices}")

# -----------------------------------
# Implementation to remove any rows that have a null value for any of the 18 kulfan parameters 
# This is caused by interrupted writing during the data generation process. 
# input_filename = r"/home/faiza/Documents/NeuralFoil/training/training_data/data_xfoil.csv"
output_filename = r"/home/faiza/Documents/NeuralFoil/training/training_data/data_xfoil_clean.csv"
# clean_csv(input_filename, output_filename)

df = pd.read_csv(output_filename)

# First 18 columns
first_18_cols = df.columns[:18]

# Check if there are any nulls
null_counts = df[first_18_cols].isnull().sum()

print("Null values per column (first 18):")
print(null_counts)

total_nulls = null_counts.sum()
if total_nulls == 0:
    print("No null values found in the first 18 columns.")
else:
    print(f"Found {total_nulls} null values in the first 18 columns.")
