import os
import neuralfoil as nf  # `pip install neuralfoil`
import aerosandbox as asb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import time
import datetime

#obtain the initial time of the process
start_time = time.time()

# Get the current date
current_date = datetime.datetime.now()
str_date = current_date.strftime("%y_%m_%d")
##########################################################################################
#######################           PARAMETERS TO MODIFY            ########################
##########################################################################################

#angle of attack (lower angle, higher angle, value to determine the number of points
reynolds = np.arange(30e4, 5.5e5, 5e4)
alpha_in = np.arange(-20, 20, 0.25)
times = []
for re in reynolds:
    Re, Alpha = np.meshgrid(re, alpha_in)

    CUT_OFF = 0.7

    #file path where the airfoils to run are located
    #Currently LucasAirfoils subfolder
    dir_path = r"C:\Users\booki\Documents\BIRD Lab\Airfoil Project\BirdAirfoils"

    output_path = r"C:\Users\booki\Documents\BIRD Lab\Airfoil Project\BirdASBXTR0.1_All"
    ##########################################################################################

    count = 0 #initialize count of airfoils ran
    total_good_results = 0

    filenames = os.listdir(dir_path) #obtain all the airfoils
    total_airfoils = len(filenames) - 1
    csv_filenames = list(filter(lambda f: f.endswith('.csv'), filenames)) # limits to csv files
    stringRe = str(int(re / 1000))

    #loop through each file name
    for file in csv_filenames:
        species_name = file.split("_")[0] + "_" + file.split("_")[1]
        bird_id = file.split("_")[2]
        pos = file.split("_")[3]
        pos = pos[:4] # Makes sure the position doesn't also include .csv, only keeps the numbers
        output_filename = str_date + "_" + species_name + "_" + bird_id + "_" + pos + "_Re_" + stringRe +"_asb.csv" #create the output file name
        file_path = r"%s/%s" % (dir_path, file)

        with open(file_path, mode='r') as f:
            csv_reader = csv.reader(f)
            data = [row for row in csv_reader]

        df_array = np.array(data).astype(float)
        # ------- Run AeroSandBox --------
        new_directory = r"C:\Users\booki\Documents\BIRD Lab\Airfoil Project"

        os.chdir(new_directory)
        airfoil = asb.Airfoil(coordinates=df_array)
        analysis = asb.XFoil(
            airfoil = airfoil, Re=re, 
            max_iter=300, 
            timeout=120,
            xfoil_command="xfoil",
            xtr_lower=0.1, xtr_upper=0.1
        )

        aero = analysis.alpha(alpha=alpha_in)

        #OBTAIN FIGURE, FROM NEURAL FOIL
        #fig, ax = plt.subplots(figsize=(6, 2))
        # np_array.draw()

        aero_out = {key: aero[key] for key in aero.keys()
                    & {'CD', 'CDp', 'CL', 'CM', 'Top_Xtr', 'Bot_Xtr', 'alpha'}}

        aero_out_keys = aero_out.keys()

        field_names = ['CD', 'CDp', 'CL', 'CM', 'Top_Xtr', 'Bot_Xtr', 'alpha']
        with open('output.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()

        #aero_out['alpha'] = alpha = Alpha.flatten()
        #aero_out['Re'] = Re = Re.flatten()
        aero_output = pd.DataFrame.from_dict(aero_out, orient='index').transpose()
        aero_output['species'] = species_name
        aero_output['Re'] = re
        aero_output = aero_output[['species', 'Re', 'CD', 'CDp', 'CL', 'CM', 'Top_Xtr', 'Bot_Xtr', 'alpha']]
        aero_output.columns = map(str.lower, aero_output.columns)     # make all the columns lowercase so its consistent with RStudio

        # output only the results that meet the threshold - WILL CUT-OFF in R
        #indices = np.where(np.array(aero_output.analysis_confidence) > CUT_OFF)[0].tolist()
        #aero_output = aero_output.loc[indices]
        # tracking the number of good results
        # total_good_results = total_good_results + len(indices)

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        output_fullname = os.path.join(output_path, output_filename) # outputs to a specific directory
        aero_output.to_csv(output_fullname)

    # obtain the difference in time from the initial to obtain the time taken to run
    print("--- %s seconds ---" % (time.time() - start_time))
    times.append(time.time() - start_time)
    print(total_good_results)

print(times)
