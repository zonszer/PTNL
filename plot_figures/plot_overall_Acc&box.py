# In[1]:

# Import necessary libraries
import glob
import re
import matplotlib.pyplot as plt
import datetime
import json
import numpy as np
current_time = datetime.datetime.now()
current_time = current_time.strftime("%m%d-%H_%M_%S")

##================set params:
path2 = '../analyze_result_temp/evalset_acc_sumlist'
# Additional specific classes to display
# additional_classes = ['crocodile_head', 'crayfish', 'anchor', 'crab', 'octopus', 'lamp']
# additional_classes = ['Apply_Eye_Makeup',
# 'Archery',
# 'Balance_Beam',
# 'Biking']
additional_classes = ['beaver',
'mouse',
'flatfish',
'otter',
'raccoon']
##================set params:

color_map = plt.cm.get_cmap("jet", len(additional_classes) + 1)
files_all_evalsetACC = sorted(glob.glob(path2 +'/*.json'), key=lambda x: int(re.findall('(\d+)', x)[-1]))
files_all_dict = {i:file.split("/")[-1] for i, file in enumerate(files_all_evalsetACC)}

#1. select the file name:
file_plot_evalsetACC = files_all_evalsetACC[163];  #can see files_all_dict #NOTE also need to set here manually, bseline ACC (PLL0)
for i, item in enumerate(files_all_evalsetACC):
    if 'SSOxfordPets-16-0-2-PLL0.3_rc_cav_beta0.2.json' in item:    #SSUCF101-16-0-2-PLL0.3_rc_cav_beta0.3.json
        print('found id:', i, ' ' +item)

with open(file_plot_evalsetACC, "r") as file:      
    data = json.load(file)
print(f'data={file_plot_evalsetACC}', f'len={len(data)}' , data)

#2. select the correspoding file name from class_acc_sumlist:
path = '../analyze_result_temp/class_acc_sumlist'
files_all = sorted(glob.glob(path +'/*.json'), key=lambda x: int(re.findall('(\d+)', x)[-1]))
files_all_names = [file.split("/")[-1] for file in files_all]
try:
    idx = files_all_names.index(file_plot_evalsetACC.split("/")[-1])      
except ValueError:
    print(f'current file name {file_plot_evalsetACC.split("/")[-1]} not match any file in {path}')

file_plot = files_all[idx]     
with open(file_plot, "r") as file:      
    data_classes = json.load(file)
print(f'data={file_plot}', f'len={len(data_classes)}' , data_classes)

#3. compute the avg acc of each class:
epoch_averages = [np.mean([epoch_data[class_name] for class_name in epoch_data]) for epoch_data in data_classes]


# %%
classes_avgAcc = epoch_averages     
valset_Acc = data
# Define the epochs based on the length of accuracies list
epochs = list(range(1, len(valset_Acc) + 1))

# Create the plot
fig = plt.figure(figsize=(17, 17))

# 1. plot the acc of the model on the whole val set during training epochs
ax1 = fig.add_subplot(311)  # 3 rows, 1 column, 1st plot
ax1.plot(epochs, valset_Acc, marker="o", linestyle=":", label=f"Accuracy")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy (%)")
ax1.set_title(f"Accuracy Variation across Epochs - Overall valset Accuracy, File name: {file_plot_evalsetACC.split('/')[-1]}")
ax1.grid()
ax1.legend()

# 2. Plot the average accuracy of all classes per epoch
ax2 = fig.add_subplot(312)  # 3 rows, 1 column, 2nd plot
ax2.plot(epochs, classes_avgAcc, marker="x", label="Average across all classes", color=color_map(len(additional_classes)))
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Average classes Accuracy (%)")
ax2.set_title("Accuracy Variation of Classes across Epochs - Average Accuracy")
ax2.grid()
ax2.legend()

# 3. additional_classes acc during training epochs:
ax3 = fig.add_subplot(313)  # 3 rows, 1 column, 3rd plot
for class_name in additional_classes:  # Added this loop
    if class_name in data_classes[0]:  # Check if the class exists in the data
        class_acc = [epoch_data[class_name] for epoch_data in data_classes]
        ax3.plot(epochs, class_acc, marker=".", label=class_name)  # Plot the class accuracy
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Accuracy (%)")
ax3.set_title("Accuracy Variation across Epochs - Additional Classes")
ax3.grid()
ax3.legend()

plt.tight_layout()
plt.show()

# %%
#use_boxplot:
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Calculate the average of 100 classes
# NOTE: fill data_PLL or data or *data_classes*:
data_box = np.array([[v for v in d.values()] for d in data_classes])  # shape: (51, 100)  

# Create a figure and axis
fig, ax = plt.subplots(figsize=(14, 8))

# Create a boxplot using the avg_data
boxplot = ax.boxplot(data_box.T, notch=True, widths=0.5)

# Create a strip plot to show data points
epochs = np.arange(1, 52)
for epoch, epoch_data in enumerate(data_box, start=1):
    y = epoch_data
    x = np.random.normal(epoch, 0.04, size=len(y))  # Add some jitter for better visibility
    ax.plot(x, y, 'r.', alpha=0.2)

# Set axis labels
ax.set_xlabel('Epochs --> ')
ax.set_ylabel('Average Values')

# Set tick marks and spacing
ax.set_xticks(epochs)
ax.xaxis.set_tick_params(rotation=90)

# Improve the readability of the plot by reducing the number of boxes shown
ax.set_xticks(np.arange(1, 52, 2))
ax.set_xticklabels(np.arange(1, 52, 2))

# Configure the grid
ax.grid(True, linestyle='--', linewidth=0.5)

# Display the boxplot
plt.show()
# %%
