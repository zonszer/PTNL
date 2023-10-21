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

current_time = datetime.datetime.now()
current_time = current_time.strftime("%m%d-%H_%M_%S")
# Step 1: Prepare separate lists
all_data_no_true = []
all_data_true = []
path = '../analyze_result_temp/logits&labels_10.14_pool'

#+++++++=========== figure5, dataset is val -> ce PLL00     #NOTE 记得需要每次手动改一下最后一个.pt name -> _test
# gt_label-cc_refine_1epoch_PLL1e-30-0.pt
# pools_dict_cc_refine_1epoch_PLL0.1-8.pt
# pred_label-cc_refine_1epoch_PLL0.3-3.pt
#+++++++=========== figure8, dataset is train -> PLL05 confidence_RC+gce_rc_PLL0.5-4.pt
gt_label_files = sorted(glob.glob(path+ '/gt_label-cc_refine_PLL0.3-*.pt'),  key=lambda x: int(re.findall('(\d+)', x)[-1]))
pred_label_files = sorted(glob.glob(path+ '/pred_label-cc_refine_PLL0.3-*.pt'),  key=lambda x: int(re.findall('(\d+)', x)[-1]))
pools_dict_files = sorted(glob.glob(path+ '/pools_dict-cc_refine_PLL0.3-*.pt'),  key=lambda x: int(re.findall('(\d+)', x)[-1]))
fig_id = 'rc_refine-pools-'
##============set params:========================

# Step 3: Load data from files into their respective lists
print(f'gt_label is {gt_label_files} \n')
print(f'pred_label is {pred_label_files} \n')
all_pred_labels = []
for file in pred_label_files:
    tensor = torch.load(file)    # Load tensor from file
    all_pred_labels.append(tensor)


all_gt_labels = []
for file in gt_label_files:
    tensor = torch.load(file)    # Load tensor from file
    all_gt_labels.append(tensor)

import utils_temp
all_pools_dicts = []
for file in pools_dict_files:
    tensor = torch.load(file)    # Load tensor from file
    all_pools_dicts.append(tensor)

# all_data_bool = all_data.bool()
# all_pred_labels = all_pred_labels.cpu().numpy()
# all_gt_labels = all_gt_labels.cpu().numpy()
print(f'all_gt_labels.shape is {len(all_gt_labels)}, {len(all_gt_labels[1])}')


# def batch_convert_dict_kvs_to_hash(list_of_dicts):
#     def convert_dict_keys_to_hash(my_dict):
#         hashable_dict = {}
#         for key, value in my_dict.items():
#             try:
#                 # Try hashing the key
#                 hash(key)
#                 if isinstance(key, torch.Tensor):
#                     raise TypeError  # Hashing tensors is not supported
#                 return my_dict  # If the key is hashable, return the original dict
#             except:
#                 # If key can't be hashed (i.e., is unhashable)
#                 if len(key.shape) > 0:  # If tensor key is not a 0-dim tensor (scalar)
#                     hashable_key = np.array2string(key.numpy())  # Convert tensor to string
#                 else:
#                     hashable_key = key.item()  # If scalar, convert to Python number
#             hashable_dict[hashable_key] = value
#         return hashable_dict
#     def convert_dict_values_to_hash(my_dict):
#         hashable_dict = {}
#         for key, value in my_dict.items():
#             try:
#                 # Try hashing the value
#                 hash(value)
#                 if isinstance(value, torch.Tensor):
#                     raise TypeError  # Hashing tensors is not supported
#                 return my_dict  # If the value is hashable, return the original dict
#             except:
#                 # If value can't be hashed (i.e., is unhashable)
#                 if hasattr(value, 'shape'):
#                     len_ = value.dim()
#                 else:
#                     len_ = len(value)
#                 if len_> 0:
#                     if isinstance(value, torch.Tensor):
#                         hashable_value = value.tolist()
#                     elif isinstance(value, list):
#                         for i in range(len(value)):
#                             value[i] = value[i].item()
#                         hashable_value = value
#                     else:
#                         raise TypeError
#                 else:
#                     hashable_value = value.item()
#             hashable_dict[key] = hashable_value
#         return hashable_dict

#     list_of_dicts = [convert_dict_keys_to_hash(my_dict) for my_dict in list_of_dicts]
#     return [convert_dict_values_to_hash(my_dict) for my_dict in list_of_dicts]

# all_gt_labels = batch_convert_dict_kvs_to_hash(all_gt_labels)
# all_pred_labels = batch_convert_dict_kvs_to_hash(all_pred_labels)

# %%
from matplotlib import ticker, colors

def calculate_correct_rate(gt_labels, pred_label):
    correct_count = 0
    for i in gt_labels:
        if i == pred_label:
            correct_count += 1
    total_count = len(gt_labels)
    if total_count == 0:
        assert correct_count == 0
        correct_rate = 0
    else:
        correct_rate = correct_count / total_count
    return correct_rate

#Step 3: Define a function to extract the data for plotting
def extract_plot_data(pools_dicts, all_gt_labels, all_pred_labels=None):
    plot_data = []
    for epoch, pool_dict in enumerate(pools_dicts):
        if epoch == 0:      # jump the first epoch 
            continue
        epoch_data = []

        for cls_idx, pool in pool_dict.items():
            gt_labels = []
            for feat_idx in pool.pool_idx:
                try:
                    gt_label = all_gt_labels[epoch][int(feat_idx.item())]    #convert dtype (may be int64) to tensor type
                    gt_labels.append(gt_label)
                except:
                    print(f'epoch is {epoch}, feat_idx is {int(feat_idx.item())}')
            
            correct_rate = calculate_correct_rate(gt_labels, pred_label=cls_idx)

            pool_data = {
                'pool_idx': cls_idx,
                'correct_rate': correct_rate,
                'pool_capacity': pool.pool_capacity,
                'pool_max_capacity': pool.pool_max_capacity,
                'unc_avg': torch.mean(pool.pool_unc).item(),
                'unc_max': pool.unc_max.item() if isinstance(pool.unc_max, torch.Tensor) else pool.unc_max
            }
            epoch_data.append(pool_data)
        plot_data.append(epoch_data)
    return plot_data


#Step 4: Define a function to create the subplots
def create_subplot(ax, epoch_data, epoch_number):
    x = np.arange(1, num_cls + 1)
    pool_capacity = [data['pool_capacity'] for data in epoch_data]
    pool_max_capacity = [data['pool_max_capacity'] for data in epoch_data]
    unc_avg = [data['unc_avg'] for data in epoch_data]
    unc_max = [data['unc_max'] for data in epoch_data]
    correct_rate = [data['correct_rate'] for data in epoch_data]
    pool_idx = [data['pool_idx'] for data in epoch_data]

    # Create a twin axes for second bar
    # ax3 = ax.twiny()
    # bars1 = ax.bar(x, pool_max_capacity, color=color_map_red.to_rgba(pool_max_capacity), alpha=0.7, width=1)
    # bars2 = ax3.bar(x, pool_capacity, color=color_map_orange.to_rgba(pool_capacity), alpha=0.7, width=1) 
    bars1 = ax.bar(x, pool_max_capacity, color='r', alpha=0.5, width=1)
    # bars2 = ax3.bar(x, pool_capacity, color='gray', alpha=0.5, width=1) 

    # Plot lines
    ax2 = ax.twinx()
    ax2.plot(x, unc_avg, color='b', linestyle='--', label='unc_avg', linewidth=1)
    # ax2.plot(x, unc_max, color='y', linestyle='--', label='unc_max', linewidth=1)
    ax2.plot(x, correct_rate, color='g', linestyle='--', label='correct_rate', linewidth=1.2)

    # Add labels on top of bars 
    for i, bar1 in enumerate(bars1):
        ax.text(bar1.get_x() + bar1.get_width() / 2, bar1.get_height(),
                str(pool_idx[i]), ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Classes', fontsize=8)
    ax.set_ylabel('Pool Capacity', fontsize=8)
    ax2.set_ylabel('Uncertainty', fontsize=8)
    ax.set_title(f'Epoch {epoch_number}', fontsize=7)

    ax.set_xticks(np.arange(1, num_cls, 1))  
    # ax.set_xticklabels(np.arange(1, num_cls, 3), fontsize=6)  # set xticks text size smaller

    ax.grid(True)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(10))
    ax.yaxis.set_minor_locator(ticker.MaxNLocator(50))
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(10))
    ax2.yaxis.set_minor_locator(ticker.MaxNLocator(50))
    return ax2  #添加返回 ax2


#---------------------settings-----------------
# Plot the subfigures
interval = 2
rows = 6
cols = 5
assert rows * cols >= len(all_pred_labels) // interval+1    #all_pred_labels or plot_data
#---------------------settings-----------------
from matplotlib import gridspec

# Extract the data for plotting
num_cls = len(set(all_gt_labels[1].values()))
plot_data = extract_plot_data(all_pools_dicts, all_gt_labels, all_pred_labels)

# Plot the subplots in one figure
max_num_subplots = len(plot_data) // interval + 1
fig = plt.figure(figsize=(cols * 8, rows * 4))
gs = gridspec.GridSpec(rows, cols, figure=fig)

axes = []         # 这个列表将存放主Y轴
secondary_axes = []  #这个列表将存放次Y轴
for i in range(max_num_subplots):
    epoch_number = i * interval - 1
    row = i // cols
    col = i % cols

    if i == 0:
        ax = fig.add_subplot(gs[row, col])
    else:
        ax = fig.add_subplot(gs[row, col], sharex=axes[0], sharey=axes[0])
    #======sub fig 2: ==========
    # ax1 = plt.subplot(gs[row, col].subgridspec(1, 1)[0])
    # ax2 = plt.subplot(gs[i // cols, i % cols].subgridspec(2, 1)[1])
    # create_subplot_new(ax1, all_pred_labels[epoch_number], all_gt_labels[epoch_number], epoch_number) 
    axes.append(ax)
    ax2 = create_subplot(ax, plot_data[epoch_number], epoch_number)
    secondary_axes.append(ax2)  # 添加ax2 到 次Y轴列表

# 共享第一个子图的次Y轴，设置其他子图的次Y轴共享
for secondary_ax in secondary_axes[1:]:
    secondary_ax.get_shared_y_axes().join(secondary_axes[0], secondary_ax)

fig.subplots_adjust(hspace=0.27, wspace=0.2)  # Adding extra space for the non-overlapping effect
plt.show()


# %%
# --------------------------plot the relationship between classes acc and CE loss and pre_num:(another verion in plotworst_class)-----------------
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
def create_subplot_new(ax1, pred_labels_epoch, gt_labels_epoch, epoch_number, dict_as_input_format=True):
    if dict_as_input_format:
        pred_val = [i[0] for i in pred_labels_epoch.values()]

        pred_indices = np.array(pred_val)   # Predicted class labels
        image_labels = np.array(list(gt_labels_epoch.values()))    # Ground truth class labels

        # Count total occurrences for each class in pred_labels and gt_labels
        class_counts_pred = Counter(pred_val)        
        class_counts_gt = Counter(gt_labels_epoch.values())   

        acc_dict = {i:0 for i in range(len(class_counts_gt))}
        for feat_idx, pred_cls in pred_labels_epoch.items():
            if pred_cls[0] == gt_labels_epoch[feat_idx]:
                acc_dict[pred_cls[0]] += 1
        # # method 1: calculate the acc by all div by 16
        # acc_dict = {class_idx: acc_dict[class_idx]/class_counts_gt[class_idx] for class_idx in range(len(acc_dict))}
        # method 2: calculate the acc by all div by the pool current capacity (HACK: should use poolCLass to calculate, use pred_labels is not right)
        acc_dict_ = {}
        for class_idx in range(len(acc_dict)):
            if acc_dict[class_idx]==0 and class_counts_pred[class_idx]==0:
                acc_dict[class_idx] = 0
            else:
                acc_dict_[class_idx] = acc_dict[class_idx]/class_counts_pred[class_idx]
        acc_dict = acc_dict_
        # print(f'epoch is {epoch_number}, acc_dict[class_idx] is {acc_dict[class_idx]}, class_counts_pred[class_idx] is {class_counts_pred[class_idx]}')
        acc_array = np.array(list(acc_dict.values()))

    else:
        # Assuming image_logits and image_labels are torch tensors
        image_logits_, image_labels_ = torch.from_numpy(image_logits), torch.from_numpy(image_labels)
        pred_indices = np.argmax(image_logits, axis=1)  #pred_labels is a dict which like {example_idx: pred class index(labels)}

    # Count occurrences for each class
    class_counts = Counter(pred_indices)        
    class_counts_true = Counter(image_labels)   #gt_labels is a dict which like {example_idx: true class index(labels)}
    if epoch_number == 43:
        print(f'class_counts_true: {class_counts_true}')
        print(f'class_counts: {class_counts}')

    # Convert keys to class labels
    class_counts_dict = {class_idx: count - class_counts_true[class_idx] for class_idx, count in class_counts.items()}
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

    # Plot the first subfigure
    # Class Counts Plot
    sc1 = ax1.scatter(class_counts[class_counts_idx], acc_array[class_counts_idx], c=ordered_colors, cmap=colormap, marker='o', alpha=0.7)
    ax1.set_title(f'Class Counts vs Class ACC epoch:{epoch_number}')
    ax1.set_xlabel('Class Counts')
    ax1.set_ylabel('Class ACC')
    ax1.grid(True)
    cbar1 = plt.colorbar(sc1, ax=ax1, ticks=np.linspace(0, 1, len(class_counts)), label='Class Index')
    cbar1.ax.tick_params(labelsize=7)  # Set the size of the legend text





# %%
