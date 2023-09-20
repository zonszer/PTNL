# In[1]:

# Import necessary libraries
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import json

current_time = datetime.datetime.now()
current_time = current_time.strftime("%m%d-%H_%M_%S")
# Step 1: Prepare separate lists
all_data_no_true = []
all_data_true = []
##================set params:
path = '../analyze_result_temp/grad_ratios_dict'    #SSCaltech101-16-0-2-PLL0.5_cc
# Number of worst-performing classes to display
num_worst_classes = 10

# Whether to compare data and data_PLL:
compare_data_PLL = True

# Additional specific classes to display
additional_classes = [ ]
##================set params:

files_all = sorted(glob.glob(path +'/*.json'), key=lambda x: int(re.findall('(\d+)', x)[-1]))

#+++++++=========== select the file name:
file_plot = files_all[2]                            #NOTE also need to set here manually
with open(file_plot, "r") as file:
    data = json.load(file)
print(file_plot, len(data))

# %%
import math
import numpy as np
import matplotlib.pyplot as plt

# Helper function to calculate the moving average
def moving_average(data, window_size):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    return (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size


# Define the grad_ratios
grad_ratios = data['prompt_learner']
filtered_data = [x for x in grad_ratios if not math.isnan(x)]

# Set the degree of smoothness (higher number => smoother curve)
smoothness = 13

# Compute the moving average on the filtered_data
smoothed_data = moving_average(np.array(filtered_data), smoothness)

# Define the epochs based on the length of smoothed_data list
epochs = list(range(smoothness//2, len(filtered_data) - smoothness//2))

# Create the plot
plt.figure(figsize=(13, 6))  # Adjust the figsize for better resolution
plt.plot(epochs, smoothed_data, marker="o", markersize=3, linestyle=":", label="Gradient Ratios")

# Label the axes and title the plot
plt.xlabel("Batch")
plt.ylabel("Gradient Ratios")
plt.title("Gradient Ratio Variation across Batches")

# Add a grid for better readability
plt.grid()

# Display the legend and plot
plt.legend()
plt.show()

# Save the plot as SVG
plt.savefig(f'Grad-{file_plot.split("/")[-1]}_{current_time}.svg')


# %%
