# In[1]:
# Import necessary libraries
import glob
import re
import torch
import joypy
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
import torch.nn as nn

current_time = datetime.datetime.now()
current_time = current_time.strftime("%m%d-%H_%M_%S")
# Step 1: Prepare separate lists
all_data_no_true = []
all_data_true = []
path = '../analyze_result_temp/logits&labels_11.11_ssucf101'

#+++++++=========== figure5, dataset is val -> ce PLL00     #NOTE 记得需要每次手动改一下最后一个.pt name -> _test
# gt_label-cc_refine_1epoch_PLL1e-30-0.pt
# pools_dict_cc_refine_1epoch_PLL0.1-8.pt
# pred_label-cc_refine_1epoch_PLL0.3-3.pt outputs_RC_RC_PLL0.3-8.pt
#+++++++=========== 11.12 figures -> outputs_CE_PLL1e-30-6 | outputs_RC_REFINE_PLL0.3-*.pt |outputs_RC_RC_PLL0.3-*.pt
gt_label_files = sorted(glob.glob(path+ '/labels_true_RC_REFINE_PLL0.3-*.pt'),  key=lambda x: int(re.findall('(\d+)', x)[-1]))
pred_label_files = sorted(glob.glob(path+ '/outputs_RC_RC_PLL0.3-*.pt'),  key=lambda x: int(re.findall('(\d+)', x)[-1]))   
# pools_dict_files = sorted(glob.glob(path+ '/pools_dict-cc_refine_PLL0.3-*.pt'),  key=lambda x: int(re.findall('(\d+)', x)[-1]))
fig_id = 'rc_rc-LOSS'
##============set params:========================

# Step 3: Load data from files into their respective lists
print(f'gt_label is {gt_label_files} \n')
print(f'pred_label is {pred_label_files} \n')
all_pred_labels = []
for file in pred_label_files:
    tensor = torch.load(file)    # Load tensor from file
    all_pred_labels.append(tensor)

all_pred_labels = torch.stack(all_pred_labels, dim=0)

all_gt_labels = []
for file in gt_label_files:
    tensor = torch.load(file)    # Load tensor from file
    all_gt_labels.append(tensor)

all_gt_labels = torch.stack(all_gt_labels, dim=0)

all_pred_labels = all_pred_labels.cpu() #torch.Size([50, 752, 47])
all_gt_labels = all_gt_labels.cpu() #torch.Size([1, 752])
print(f'all_gt_labels.shape is {all_gt_labels.shape}')


# %%
from matplotlib import ticker, colors

# Step 2: Modify the calculate_correct_rate function
def calculate_correct_rate(gt_labels, pred_labels):
    correct_count = (gt_labels == pred_labels).sum().item()
    total_count = len(gt_labels)
    correct_rate = correct_count / total_count if total_count > 0 else 0
    return correct_rate

# Step 3: Modify the extract_plot_data function
def extract_plot_data(all_gt_labels, image_logits_):
    epoch_data = []
    all_gt_labels = all_gt_labels.squeeze(0)
    for epoch in range(image_logits_.shape[0]):
        pred_labels = torch.argmax(image_logits_[epoch], dim=1)
        cls_countNum = {}
        for class_idx in range(image_logits_[epoch].shape[1]):
            cls_countNum[class_idx] = (pred_labels == class_idx).sum().item()
        correct_rate = calculate_correct_rate(all_gt_labels, pred_labels)

        # Calculate cross entropy loss:
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        ce_values = loss_fn(image_logits_[epoch], pred_labels)    #NOTE choose all_gt_labels or pred_labels

        # Calculate avg CE loss for each class:
        class_ce_dict = {}
        for class_idx in range(image_logits_[epoch].shape[1]):
            class_indices = (all_gt_labels == class_idx)
            class_ce_dict[class_idx] = ce_values[class_indices].mean().item()

        _, avg_ce_values = zip(*sorted(class_ce_dict.items()))
        avg_ce_values = np.array(avg_ce_values)

        pool_data = {
            'overall_ACC': correct_rate,
            'cls_lossAvg': avg_ce_values,
            'cls_countNum': cls_countNum.values(),
            'pred_labels': pred_labels.numpy(),
        }
        epoch_data.append(pool_data)
    return epoch_data

#Step 4: Define a function to create the subplots
# --------------------------plot the relationship between classes acc and CE loss and pre_num:(another verion in plotworst_class)-----------------
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch

def create_subplot_new(ax1, ax2,
                       pred_labels_epoch, gt_labels_epoch, avg_ce_values,
                       epoch_number, dict_as_input_format=False):
    pred_indices = pred_labels_epoch   # Predicted class labels
    image_labels = gt_labels_epoch    # Ground truth class labels

    # Count total occurrences for each class in pred_labels and gt_labels
    class_counts_pred = Counter(pred_indices)        
    class_counts_gt = Counter(image_labels)   

    acc_dict = {i:0 for i in range(len(class_counts_gt))}
    for pred_cls, true_cls in zip(pred_indices, image_labels):
        if pred_cls == true_cls:
            acc_dict[pred_cls] += 1

    acc_dict_ = {i:0 for i in range(len(class_counts_gt))}
    for class_idx in range(len(acc_dict)):
        if acc_dict[class_idx]==0 and class_counts_pred[class_idx]==0:
            acc_dict[class_idx] = 0
        else:
            # acc_dict_[class_idx] = acc_dict[class_idx]/class_counts_pred[class_idx]
            acc_dict_[class_idx] = acc_dict[class_idx]/class_counts_gt[class_idx]
    acc_dict = acc_dict_

    acc_array = np.array(list(acc_dict.values()))

    # Count occurrences for each class
    class_counts = Counter(pred_indices)        
    class_counts_true = Counter(image_labels)
    if epoch_number == 43:
        print(f'class_counts_true: {class_counts_true}')
        print(f'class_counts: {class_counts}')

    # Convert keys to class labels
    class_counts_dict = {i:0 for i in range(len(class_counts_gt))}
    for class_idx, count in class_counts.items():
        class_counts_dict[class_idx] = count - class_counts_true[class_idx]
    class_counts = sorted(class_counts_dict.items())
    class_counts = np.array(class_counts)[:,1]
    class_counts_idx = np.argsort(class_counts)

    # Create a custom colormap
    # Step 1: Define color groups based on data length
    split_num = 6
    color_opt = ['orange', 'blue', 'green', 'red', 'purple', 'pink', 'brown', 'gray', 'olive', 'cyan']
    for i in range(split_num):
        if i == 0:
            colors = [color_opt[i]] * (len(class_counts)//split_num)
        else:
            colors = colors + [color_opt[i]] * (len(class_counts)//split_num)
    # Step 2: Create a colormap
    colormap = mcolors.ListedColormap(colors)
    # Step 3: Create an array of colors for points in the scatter plot
    color_array = np.linspace(0, 1, num=len(class_counts))
    # Step 4: Assign colors to data points based on their order
    ordered_colors = color_array

    # Class Counts Plot
    sc1 = ax1.scatter(class_counts[class_counts_idx], acc_array[class_counts_idx], c=ordered_colors, cmap=colormap, marker='o', alpha=0.6)
    ax1.set_title(f'Class Counts vs Class ACC epoch:{epoch_number}')
    ax1.set_xlabel('Class Counts')
    ax1.set_ylabel('Class ACC')
    ax1.grid(True)
    cbar1 = plt.colorbar(sc1, ax=ax1, ticks=np.linspace(0, 1, len(class_counts)), label='Class Index')
    cbar1.ax.tick_params(labelsize=7)  # Set the size of the legend text

    avg_ce_values_idx = np.argsort(avg_ce_values)

    color_class_idx_order = ordered_colors[np.argsort(class_counts_idx)]
    sc2 = ax2.scatter(avg_ce_values[avg_ce_values_idx], acc_array[avg_ce_values_idx], c=color_class_idx_order[avg_ce_values_idx], cmap=colormap, marker='o', alpha=0.6)
    ax2.set_title(f'Avg CE vs Avg ACC, classes_collect_method: pred_indices, epoch:{epoch_number}')
    ax2.set_xlabel('Class Avg CE')
    ax2.set_ylabel('Class ACC')
    ax2.grid(True)
    cbar2 = plt.colorbar(sc2, ax=ax2, ticks=np.linspace(0, 1, len(class_counts)), label='Class Index')
    cbar2.ax.tick_params(labelsize=7)  # Set the size of the legend text

#--------------------------------settings----------------------------
# Plot the subfigures
interval = 2
rows = 6
cols = 5
assert rows * cols >= len(all_pred_labels) // interval+1    #all_pred_labels or plot_data
#--------------------------------settings----------------------------
from matplotlib import gridspec

# Extract the data for plotting
plot_data = extract_plot_data(all_gt_labels, all_pred_labels)

# Plot the subplots in one figure
max_num_subplots = len(plot_data) // interval + 1
fig = plt.figure(figsize=(cols * 8, rows * 8))      #NOTE adjust here to adjust the size of GridSpec object 
gs = gridspec.GridSpec(rows, cols, figure=fig)

axes = []         # 这个列表将存放主Y轴
secondary_axes = []  #这个列表将存放次Y轴
for i in range(max_num_subplots):
    epoch_number = i * interval 
    row = i // cols
    col = i % cols

    # Define a nested GridSpec object
    gs_i = gs[row, col].subgridspec(2, 1, height_ratios=[2, 2])

    # if i == 0:
    #     ax = fig.add_subplot(gs_i[0])
    # else:
    #     ax = fig.add_subplot(gs_i[0], sharex=axes[0], sharey=axes[0])
    #======sub fig 2: ==========
    ax1 = fig.add_subplot(gs_i[0])
    ax2 = fig.add_subplot(gs_i[1])
    create_subplot_new(ax1, ax2,
                       pred_labels_epoch=plot_data[epoch_number]['pred_labels'], 
                       gt_labels_epoch=all_gt_labels.squeeze(0).numpy(), 
                       avg_ce_values=plot_data[epoch_number]['cls_lossAvg'],
                       epoch_number=epoch_number) 
    axes.append(ax1)
    secondary_axes.extend([ax1, ax2])  # 添加ax2 到 次Y轴列表
    # axes.append(ax)
    # ax2 = create_subplot(ax, plot_data[epoch_number], epoch_number)
    # secondary_axes.append(ax2)  # 添加ax2 到 次Y轴列表

# 共享第一个子图的次Y轴，设置其他子图的次Y轴共享
for secondary_ax in secondary_axes[1:]:
    secondary_ax.get_shared_y_axes().join(secondary_axes[0], secondary_ax)

fig.subplots_adjust(hspace=0.26, wspace=0.2)  # Increase the vertical spacing
plt.show()


# %%

import matplotlib.pyplot as plt

# Your data
data_PLL03_rc_rc = [0.19680851063829788, 0.26063829787234044, 0.2752659574468085, 0.3058510638297872, 0.3377659574468085, 0.37632978723404253, 0.40824468085106386, 0.42154255319148937, 0.4521276595744681, 0.43882978723404253, 0.46675531914893614, 0.4973404255319149, 0.511968085106383, 0.5332446808510638, 0.5531914893617021, 0.5784574468085106, 0.5718085106382979, 0.5944148936170213, 0.5824468085106383, 0.5930851063829787, 0.5930851063829787, 0.613031914893617, 0.6170212765957447, 0.6090425531914894, 0.6037234042553191, 0.6436170212765957, 0.6223404255319149, 0.6303191489361702, 0.6196808510638298, 0.6396276595744681, 0.6276595744680851, 0.6409574468085106, 0.6409574468085106, 0.6117021276595744, 0.6289893617021277, 0.6090425531914894, 0.6263297872340425, 0.636968085106383, 0.636968085106383, 0.6329787234042553, 0.6223404255319149, 0.6356382978723404, 0.6422872340425532, 0.6409574468085106, 0.6356382978723404, 0.636968085106383, 0.6422872340425532, 0.6409574468085106, 0.6422872340425532]
data_PLL03_rc_refine = [0.1981382978723404, 0.3257978723404255, 0.37101063829787234, 0.45345744680851063, 0.4800531914893617, 0.5066489361702128, 0.4973404255319149, 0.5106382978723404, 0.5372340425531915, 0.5558510638297872, 0.5651595744680851, 0.5691489361702128, 0.5811170212765957, 0.6063829787234043, 0.6077127659574468, 0.6156914893617021, 0.625, 0.6263297872340425, 0.6343085106382979, 0.6316489361702128, 0.6449468085106383, 0.6329787234042553, 0.6303191489361702, 0.6329787234042553, 0.6396276595744681, 0.6303191489361702, 0.6329787234042553, 0.6436170212765957, 0.6422872340425532, 0.6476063829787234, 0.6462765957446809, 0.648936170212766, 0.6462765957446809, 0.6476063829787234, 0.660904255319149, 0.6542553191489362, 0.651595744680851, 0.6476063829787234, 0.6422872340425532, 0.651595744680851, 0.6502659574468085, 0.651595744680851, 0.6449468085106383, 0.6476063829787234, 0.6476063829787234, 0.6449468085106383, 0.6449468085106383, 0.6462765957446809, 0.6449468085106383]
data_PLL00_CE = [0.2393617021276596, 0.5199468085106383, 0.5638297872340425, 0.6183510638297872, 0.6356382978723404, 0.6476063829787234, 0.6343085106382979, 0.648936170212766, 0.6914893617021277, 0.711436170212766, 0.7433510638297872, 0.7446808510638298, 0.7367021276595744, 0.7513297872340425, 0.7699468085106383, 0.7566489361702128, 0.7712765957446809, 0.7872340425531915, 0.761968085106383, 0.8031914893617021, 0.8085106382978723, 0.8058510638297872, 0.8271276595744681, 0.8231382978723404, 0.8098404255319149, 0.8164893617021277, 0.8204787234042553, 0.8178191489361702, 0.8351063829787234, 0.836436170212766, 0.8497340425531915, 0.8523936170212766, 0.8417553191489362, 0.8603723404255319, 0.8563829787234043, 0.8617021276595744, 0.8803191489361702, 0.8643617021276596, 0.8696808510638298, 0.8789893617021277, 0.8856382978723404, 0.8763297872340425, 0.886968085106383, 0.8882978723404256, 0.8856382978723404, 0.8936170212765957, 0.8896276595744681, 0.8922872340425532, 0.886968085106383]

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the data
ax.plot(data_PLL03_rc_rc, label='data-PLL03_rc_rc')
ax.plot(data_PLL03_rc_refine, label='data-PLL03_rc_refine')
ax.plot(data_PLL00_CE, label='data-PLL00_CE')

# Set labels
ax.set_xlabel('Epoch')
ax.set_ylabel('ACC')
ax.set_title('ACC of each epoch')

# Add a legend
ax.legend()

# Show the plot
plt.show()

# %%
