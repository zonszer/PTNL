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
path = '../analyze_result_temp/class_acc_sumlist'
# Number of worst-performing classes to display
num_worst_classes = 5

# Whether to compare data and data_PLL:
compare_data_PLL = True

# Additional specific classes to display
additional_classes = [ ]
##================set params:

files_all = sorted(glob.glob(path +'/*.json'), key=lambda x: int(re.findall('(\d+)', x)[-1]))

#+++++++=========== select the file name:
file_plot = files_all[0]                            #NOTE also need to set here manually, bseline ACC (PLL0)
with open(file_plot, "r") as file:
    data = json.load(file)
print(len(data), data )

if compare_data_PLL == True:
    file_plot_PLL = files_all[1]                    #NOTE also need to set here manually
    with open(file_plot_PLL, "r") as file:
        data_PLL = json.load(file)
    print(len(data_PLL), data_PLL )

# %%
import matplotlib.pyplot as plt
import numpy as np

def compute_class_avg(data, class_name):
    class_accuracies = [item[class_name] for item in data]
    return np.mean(class_accuracies)
def compute_avg_gap(data, data_PLL, class_name):
    '''gap is the difference between data and data_PLL'''
    gaps = data[class_name] - data_PLL[class_name]
    return gaps

# check the data:
class_num = len(data[0])
epoch_num = len(data)
if compare_data_PLL == True:
    assert len(data_PLL[0]) == class_num
    assert len(data_PLL) == epoch_num
    display_classes = "largest_gap_classes + additional_classes"        #other opt: "worst_classes + additional_classes" , "largest_gap_classes + additional_classes"
else:
    display_classes = "worst_classes + additional_classes"        #other opt: "worst_classes + additional_classes" , "largest_gap_classes + additional_classes"


# Define the epochs and create a color map
epochs = list(range(1, epoch_num + 1))
color_map = plt.cm.get_cmap("jet", num_worst_classes + len(additional_classes) + 1)

# Calculate class average accuracies and average gaps between data and data_PLL
class_averages = {class_name: compute_class_avg(data, class_name) for class_name in data[0].keys()}
sorted_class_averages = sorted(class_averages.items(), key=lambda x: x[1])
if compare_data_PLL == True:
    class_averages_PLL = {class_name: compute_class_avg(data_PLL, class_name) for class_name in data_PLL[0].keys()}
    sorted_class_averages_PLL = sorted(class_averages_PLL.items(), key=lambda x: x[1])
    class_avg_gaps = {class_name: compute_avg_gap(class_averages, class_averages_PLL, class_name) for class_name in data[0].keys()}
    sorted_class_gaps = sorted(class_avg_gaps.items(), key=lambda x: x[1], reverse=True)
    # sorted_class_gaps = sorted(class_avg_gaps.items(), key=lambda x: x[1], reverse=False)


if compare_data_PLL == True:
    largest_gap_classes = [item[0] for item in sorted_class_gaps[:num_worst_classes]]
    print(f'largest_gap_classes are: {largest_gap_classes}')
    print(f'their AVG gap (data-data_PLL) are: {sorted_class_gaps[:num_worst_classes]}')
else:
    # Get the worst-performing classes and classes with the largest gap
    worst_classes = [item[0] for item in sorted_class_averages[:num_worst_classes]]
    print(f'worst_classes are: {worst_classes}')
    print(f'worst_classes are: {sorted_class_averages[:num_worst_classes]}')
display_classes = eval(display_classes)

# Compute the averages for all classes at each epoch
epoch_averages = [np.mean([epoch_data[class_name] for class_name in epoch_data]) for epoch_data in data]

# Create a larger figure for better readability
fig, ax = plt.subplots(figsize=(17, 10))


if compare_data_PLL == True:
    for idx, class_name in enumerate(display_classes, start=1):
        class_accuracies = [item[class_name] - item_PLL[class_name] for item, item_PLL in zip(data, data_PLL)]
        ax.plot(epochs, class_accuracies, marker="o", label=class_name, color=color_map(idx - 1))
else:
    for idx, class_name in enumerate(display_classes, start=1):
        class_accuracies = [item[class_name] for item in data]
        ax.plot(epochs, class_accuracies, marker="o", label=class_name, color=color_map(idx - 1))

# Plot the average accuracy of all classes per epoch
ax.plot(epochs, epoch_averages, marker="x", label="Average across all classes", color=color_map(len(display_classes)))

ax.set_xticks(epochs)
ax.set_xlabel("Epoch")
ax.set_ylabel("Class Accuracy")
ax.set_title("Class Accuracy per Epoch for Classes with Largest Gap")
ax.grid()

# Place the legend outside the plot area in a separate space
legend = ax.legend(
    ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True
)
fig.colorbar(
    plt.cm.ScalarMappable(cmap=color_map), ax=ax, label="Classes", ticks=[*range(1, epoch_num + 1, 10)]
)

# Adjust the plot layout to accommodate the external legend
plt.subplots_adjust(bottom=0.3)
plt.show()
# if compare_data_PLL == True:
#     plt.savefig(f'ACC-{file_plot_PLL.split("/")[-1]}_{current_time}.svg')
# else:
#     plt.savefig(f'ACC-{file_plot.split("/")[-1]}_{current_time}.svg')

# %%

# import matplotlib.pyplot as plt
# # Define the list of accuracies
# accuracies = [
#     78.17, 87.14, 91.09, 91.63, 91.33, 90.72, 91.02, 91.15, 91.27, 91.45,
#     91.21, 90.84, 90.18, 90.66, 90.54, 90.60, 90.12, 89.45, 89.33, 89.81,
#     90.24, 89.99, 89.93, 89.75, 89.87, 89.69
# ]

# # Define the epochs based on the length of accuracies list
# epochs = list(range(1, len(accuracies) + 1))

# # Create the plot
# plt.figure(figsize=(10, 5))
# plt.plot(epochs, accuracies, marker="o", linestyle=":", label="Accuracy")

# # Label the axes and title the plot
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy (%)")
# plt.title("Accuracy Variation across Epochs")

# # Add a grid for better readability
# plt.grid()

# # Display the legend and plot
# plt.legend()
# plt.show()


# %%
