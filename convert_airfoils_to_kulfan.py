import os
import neuralfoil as nf  # `pip install neuralfoil`
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import time
import datetime
import aerosandbox as asb
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters, get_kulfan_coordinates

#obtain the initial time of the process
start_time = time.time()

# Get the current date
current_date = datetime.datetime.now()
str_date = current_date.strftime("%y_%m_%d")

#Path to all the airofils 
dir_path = r"C:\Users\booki\Documents\BIRD Lab\Airfoil Project\Airfoils2Run"
output_path = r"C:\Users\booki\Documents\BIRD Lab\Airfoil Project"


filenames = os.listdir(dir_path) #obtain all the airfoils
total_airfoils = len(filenames) - 1
csv_filenames = list(filter(lambda f: f.endswith('.csv'), filenames)) # limits to csv files

if not os.path.exists(output_path):
    os.mkdir(output_path)

output_filename = str_date + "_bird_airfoils_kulfan_parameters.csv" #one CSV for all airfoils 
output_fullname = os.path.join(output_path, output_filename)

parameters_df = []

#loop through each file name
for file in csv_filenames:
    # used different index numbers for the
    # non live airfoils bc they had dates too
    species_name = file.split("_")[3] + "_" + file.split("_")[4]
    bird_id = file.split("_")[5]
    pos = file.split("_")[6]
    pos = pos[:4] # Makes sure the position doesn't also include .csv, only keeps the numbers
    #output_filename = str_date + "_" + species_name + "_" + bird_id + "_" + pos + "_af_gen2_1024.csv" #create the output file name

    file_path = r"%s/%s" % (dir_path, file)

    with open(file_path, mode='r') as f:
        csv_reader = csv.reader(f)
        data = [row for row in csv_reader]

    df_array = np.array(data).astype(float)
    kulfan_param = get_kulfan_parameters(df_array, n_weights_per_side=8)

    #Create an object row with airfoil name data and parameters
    row = {
        "species": species_name, 
        "bird_id": bird_id,
        "pos": pos,
    }
    #add lower weights to the row
    for i, weight in enumerate(kulfan_param["lower_weights"]):
        row[f"lower_weights_{i}"] = weight
    #add upper weights to the row
    for i, weight in enumerate(kulfan_param["upper_weights"]):
            row[f"upper_weights_{i}"] = weight
    row["LE_weight"] = kulfan_param["leading_edge_weight"]
    row["TE_thickness"] = kulfan_param["TE_thickness"]

    parameters_df.append(row)

#Convert to dataframe and then write to a .csv 
parameters_df = pd.DataFrame(parameters_df)
parameters_df.to_csv(output_fullname, index=False)

#Plot one airfoil to check that it is correct. 
# plt.figure()
# coords = get_kulfan_coordinates(lower_weights=kulfan_param["lower_weights"], 
#                                 upper_weights=kulfan_param["upper_weights"],
#                                 leading_edge_weight=kulfan_param["leading_edge_weight"],
#                                 TE_thickness=kulfan_param["TE_thickness"])
# kulfan_x = coords[:,0]
# kulfan_y = coords[:,1]
# plt.plot(kulfan_x, kulfan_y)
# plt.axis("equal")
# plt.show()

print(f"Saved {len(parameters_df)} airfoils")
print(f"Output: {output_fullname}")

# obtain the difference in time from the initial to obtain the time taken to run
print("--- %s seconds ---" % (time.time() - start_time))
