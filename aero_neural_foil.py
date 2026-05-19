import os
import neuralfoil as nf  # `pip install neuralfoil`
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import time
import datetime

# Code for analyzing all bird airfoils. 
# Remember bird airfoil file name is YY_MM_DD_Species_name_BirdID_pos_coords.csv

#obtain the initial time of the process
start_time = time.time()

# Get the current date
current_date = datetime.datetime.now()
str_date = current_date.strftime("%y_%m_%d")
##########################################################################################
#######################           PARAMETERS TO MODIFY            ########################
##########################################################################################

#angle of attack (lower angle, higher angle, value to determine the number of points
re = np.arange(5e4, 5.5e5, 5e4)
alpha = np.arange(-20, 20, 0.25)

Re, Alpha = np.meshgrid(re, alpha)

CUT_OFF = 0.7

#file path where the airfoils to run are located
# #Currently LucasAirfoils subfolder
# dir_path = r"C:\Users\booki\Documents\BIRD Lab\Airfoil Project\Airfoils2Run"

# output_path = r"C:\Users\booki\Documents\BIRD Lab\Airfoil Project\BirdAvianFoilXTR0.1_V3"

dir_path = r"C:\Users\booki\Documents\BIRD Lab\Airfoil Project\Airfoils2Run"
output_path = r"C:\Users\booki\Documents\BIRD Lab\Airfoil Project\Bird_XTR0.1_Avian_gen2_256"
##########################################################################################

count = 0 #initialize count of airfoils ran
total_good_results = 0

filenames = os.listdir(dir_path) #obtain all the airfoils
total_airfoils = len(filenames) - 1
csv_filenames = list(filter(lambda f: f.endswith('.csv'), filenames)) # limits to csv files

if not os.path.exists(output_path):
    os.mkdir(output_path)

#loop through each file name
for file in csv_filenames:
    # used different index numbers for the
    # non live airfoils bc they had dates too
    species_name = file.split("_")[3] + "_" + file.split("_")[4]
    bird_id = file.split("_")[5]
    pos = file.split("_")[6]
    pos = pos[:4] # Makes sure the position doesn't also include .csv, only keeps the numbers
    output_filename = str_date + "_" + species_name + "_" + bird_id + "_" + pos + "_af_gen2_256.csv" #create the output file name
    
    output_fullname = os.path.join(output_path, output_filename)

    # ---- SKIP IF ALREADY WRITTEN ----
    if os.path.exists(output_fullname):
        print(f"Skipping (already exists): {output_filename}")
        continue

    file_path = r"%s/%s" % (dir_path, file)

    with open(file_path, mode='r') as f:
        csv_reader = csv.reader(f)
        data = [row for row in csv_reader]

    df_array = np.array(data).astype(float)
    # ------- Run Neural Foil --------
    aero = nf.get_aero_from_coordinates(
        coordinates=df_array,
        alpha=Alpha.flatten(),
        Re=Re.flatten(),
        xtr_upper=0.1,  # Location of a forced top-side BL trip, as a fraction of chord
        xtr_lower=0.1,  # Location of a forced bottom-side BL trip, as a fraction of chord
        model_size="avian-gen2-256",  # Optionally, specify your model size.
    )

    #OBTAIN FIGURE, FROM NEURAL FOIL
    #fig, ax = plt.subplots(figsize=(6, 2))
    # np_array.draw()

    aero_out = {key: aero[key] for key in aero.keys()
                & {'analysis_confidence', 'CL', 'CD', 'CM', 'Top_Xtr', 'Bot_Xtr'}}

    aero_out_keys = aero_out.keys()

    field_names = ['analysis_confidence', 'CL', 'CD', 'CM', 'Top_Xtr', 'Bot_Xtr']
    with open('output.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()

    aero_out['alpha'] = alpha = Alpha.flatten()
    aero_out['Re'] = Re = Re.flatten()
    aero_output = pd.DataFrame.from_dict(aero_out, orient='index').transpose()
    aero_output['species'] = species_name
    aero_output = aero_output[['species', 'alpha', 'Re', 'analysis_confidence', 'CL', 'CD', 'CM', 'Top_Xtr', 'Bot_Xtr']]
    aero_output.columns = map(str.lower, aero_output.columns)     # make all the columns lowercase so its consistent with RStudio

    # output only the results that meet the threshold - WILL CUT-OFF in R
    #indices = np.where(np.array(aero_output.analysis_confidence) > CUT_OFF)[0].tolist()
    #aero_output = aero_output.loc[indices]
    # tracking the number of good results
    # total_good_results = total_good_results + len(indices)

    # if not os.path.exists(output_path):
    #     os.mkdir(output_path)

    output_fullname = os.path.join(output_path, output_filename) # outputs to a specific directory
    aero_output.to_csv(output_fullname)

# obtain the difference in time from the initial to obtain the time taken to run
print("--- %s seconds ---" % (time.time() - start_time))
print(total_good_results)
