import os
import re
import matplotlib.pyplot as plt
import numpy as np

#Set folder path
FOLDER_PATH = r"/home/faiza/Documents/NeuralFoil/avian_gen2_improvement_testing"


def parse_log_file(filepath):
    data = {}

    summary_pattern = re.compile(
        r"Epoch:\s*(\d+)\s*\|\s*Train Loss:\s*([0-9.eE+-]+)\s*\|\s*Test Loss:\s*([0-9.eE+-]+)"
    )

    time_pattern = re.compile(
        r"Duration of Epoch:\s*(\d+)\s*was\s*([0-9.eE+-]+)\s*seconds"
    )

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:

            summary_match = summary_pattern.search(line)
            if summary_match:
                e = int(summary_match.group(1))
                if e not in data:
                    data[e] = {}
                data[e]["train"] = float(summary_match.group(2))
                data[e]["test"] = float(summary_match.group(3))

            time_match = time_pattern.search(line)
            if time_match:
                e = int(time_match.group(1))
                if e not in data:
                    data[e] = {}
                data[e]["time"] = float(time_match.group(2))

    # Build aligned lists
    epochs, train_losses, test_losses, times = [], [], [], []

    for e in sorted(data.keys()):
        if "train" in data[e] and "test" in data[e] and "time" in data[e]:
            epochs.append(e)
            train_losses.append(data[e]["train"])
            test_losses.append(data[e]["test"])
            times.append(data[e]["time"])

    return epochs, train_losses, test_losses, times


def plot_losses(epochs, train_losses, test_losses, title, output_path):
    #Plot of Epoch vs Train Loss and Test Loss for each log file
    plt.figure()

    #Plot both curves on same graph
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss")

    #Labels
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xlim(left=0) #Aligns the 0 x-axis to the y-axis
    plt.title(title + " (Loss vs. Epoch)")
    plt.legend(fontsize=8)
    plt.legend(loc='upper right')
    plt.grid()

    #Save image to disk
    plt.savefig(output_path)
    plt.close()


def plot_time(epochs, times, title, output_path):
    #Plot of Epoch vs Time for each log file
    if not times:
        print(f"Skipping time plot for {title} (no time data)")
        return

    #Ensure same length before plotting
    min_len = min(len(epochs), len(times))

    plt.figure()
    plt.plot(epochs[:min_len], times[:min_len], label="Time (s)")
    plt.xlabel("Epoch")
    plt.ylabel("Time (s)")
    plt.xlim(left=0) #Aligns the 0 x-axis to the y-axis
    plt.title(title + " (Time vs. Epoch)")
    plt.legend(fontsize=8)
    plt.legend(loc='upper right')
    plt.grid()
    plt.savefig(output_path)
    plt.close()


def process_folder(folder_path):
    """
    Main driver:
    - Loops through all .log files
    - Generates individual plots
    - Builds combined plots across all files
    """

    # Store all runs for combined plots
    all_train_data = []
    all_test_data = []

    #Loop through files in folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".log"): #Only selects log files

            filepath = os.path.join(folder_path, filename)
            print(f"Processing: {filename}")

            #Extract data from log
            epochs, train_losses, test_losses, times = parse_log_file(filepath)

            #Skip files with no usable data
            if len(epochs) == 0:
                print(f"Skipping {filename} (no valid data found)")
                continue

            #INDIVIDUAL PLOTS

            #Loss plot (Train and Test)
            loss_output = os.path.join(folder_path, filename.replace(".log", "_Loss_Plot.png")) #names the file
            plot_losses(epochs, train_losses, test_losses, filename, loss_output)

            #Time plot
            time_output = os.path.join(folder_path, filename.replace(".log", "_Time_Plot.png"))
            plot_time(epochs, times, filename, time_output)
            print(f"{filename} took an average of {np.mean(times)} seconds per epoch")

            #Save data for combined plots
            all_train_data.append((filename, epochs, train_losses))
            all_test_data.append((filename, epochs, test_losses))

    #COMBINED PLOTS

    #Train Loss plots for all log files combined
    plt.figure()
    for name, epochs, train_losses in all_train_data:
        plt.plot(epochs, train_losses, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.xlim(left=0) #Aligns the 0 x-axis to the y-axis
    plt.title("Combined Train Loss")
    plt.legend(fontsize=8)
    plt.legend(loc='upper right')
    plt.grid()
    plt.savefig(os.path.join(folder_path, "Combined_Train_Loss_Plot.png"))
    plt.close()

    #Test Loss plots for all log files combined
    plt.figure()
    for name, epochs, test_losses in all_test_data:
        plt.plot(epochs, test_losses, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Test Loss")
    plt.xlim(left=0) #Aligns the 0 x-axis to the y-axis
    plt.title("Combined Test Loss")
    plt.legend(fontsize=8)
    plt.legend(loc='upper right')
    plt.grid()
    plt.savefig(os.path.join(folder_path, "Combined_Test_Loss_Plot.png"))
    plt.close()

    print("Done.")


#Entry point (runs script)
if __name__ == "__main__":
    process_folder(FOLDER_PATH)