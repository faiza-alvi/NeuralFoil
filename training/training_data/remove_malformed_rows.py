import numpy as np
import pandas as pd
from scipy.special import comb
import polars as pl

import csv

def find_longest_rows(filename):
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

# Usage
file_name = r"C:\Users\booki\Documents\BIRD Lab\Airfoil Project\data_xfoil_final_K9_clean.csv"  # Replace with your CSV file name
row_indices, length = find_longest_rows(file_name)

print(f"The longest rows have {length} columns ")
print(f"The longest rows are at indices: {row_indices}")

# import csv

# def remove_longest_rows(input_file, output_file):
#     # First pass: find max length
#     max_length = 0
#     with open(input_file, 'r', newline='') as f_in:
#         reader = csv.reader(f_in)
#         for row in reader:
#             max_length = max(max_length, len(row))

#     # Second pass: write only rows that are NOT longest
#     with open(input_file, 'r', newline='') as f_in, \
#          open(output_file, 'w', newline='') as f_out:
#         reader = csv.reader(f_in)
#         writer = csv.writer(f_out)

#         for row in reader:
#             if len(row) != max_length:
#                 writer.writerow(row)

#     return max_length

# # Usage
# input_file = r"C:\Users\booki\Documents\BIRD Lab\Airfoil Project\data_xfoil_final_K9.csv"  # Replace with your CSV file name
# output_file = r"C:\Users\booki\Documents\BIRD Lab\Airfoil Project\data_xfoil_final_K9_clean.csv"  # Replace with your CSV file name
# max_len = remove_longest_rows(input_file, output_file)

# print(f"Removed rows with {max_len} columns. New file saved as {output_file}")

