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
gt_label_files = sorted(glob.glob(path+ '/gt_label-cc_refine_1epoch_PLL0.3-*.pt'),  key=lambda x: int(re.findall('(\d+)', x)[-1]))
pred_label_files = sorted(glob.glob(path+ '/pred_label-cc_refine_1epoch_PLL0.3-*.pt'),  key=lambda x: int(re.findall('(\d+)', x)[-1]))
pools_dict_files = sorted(glob.glob(path+ '/pools_dict-cc_refine_1epoch_PLL0.3-*.pt'),  key=lambda x: int(re.findall('(\d+)', x)[-1]))
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
print(f'all_data_no_true.shape is {len(all_gt_labels)}, {len(all_gt_labels[1])}')


def batch_convert_dict_kvs_to_hash(list_of_dicts):
    def convert_dict_keys_to_hash(my_dict):
        hashable_dict = {}
        for key, value in my_dict.items():
            try:
                # Try hashing the key
                hash(key)
                if isinstance(key, torch.Tensor):
                    raise TypeError  # Hashing tensors is not supported
                return my_dict  # If the key is hashable, return the original dict
            except:
                # If key can't be hashed (i.e., is unhashable)
                if len(key.shape) > 0:  # If tensor key is not a 0-dim tensor (scalar)
                    hashable_key = np.array2string(key.numpy())  # Convert tensor to string
                else:
                    hashable_key = key.item()  # If scalar, convert to Python number
            hashable_dict[hashable_key] = value
        return hashable_dict
    def convert_dict_values_to_hash(my_dict):
        hashable_dict = {}
        for key, value in my_dict.items():
            try:
                # Try hashing the value
                hash(value)
                if isinstance(value, torch.Tensor):
                    raise TypeError  # Hashing tensors is not supported
                return my_dict  # If the value is hashable, return the original dict
            except:
                # If value can't be hashed (i.e., is unhashable)
                if len(value) > 0:
                    if isinstance(value, torch.Tensor):
                        hashable_value = value.tolist()
                    elif isinstance(value, list):
                        for i in range(len(value)):
                            value[i] = value[i].item()
                        hashable_value = value
                    else:
                        raise TypeError
                else:
                    hashable_value = value.item()
            hashable_dict[key] = hashable_value
        return hashable_dict

    list_of_dicts = [convert_dict_keys_to_hash(my_dict) for my_dict in list_of_dicts]
    return [convert_dict_values_to_hash(my_dict) for my_dict in list_of_dicts]

all_gt_labels = batch_convert_dict_kvs_to_hash(all_gt_labels)
all_pred_labels = batch_convert_dict_kvs_to_hash(all_pred_labels)

# %%
from matplotlib import ticker, colors

def calculate_correct_rate(gt_labels, pred_label):
    correct_count = sum(gt_labels == pred_label)
    total_count = len(gt_labels)
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
                'unc_max': pool.unc_max.item()
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

    # Normalized color mappings
    clr = colors.Normalize(min(pool_max_capacity), max(pool_max_capacity))
    color_map_orange = plt.cm.ScalarMappable(cmap='Oranges', norm=colors.Normalize(vmin=0, vmax=max(pool_capacity)))
    color_map_red = plt.cm.ScalarMappable(cmap='Reds', norm=clr)
    
    # Create a twin axes for second bar
    # ax3 = ax.twiny()
    # Plot bars
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



# Extract the data for plotting
num_cls = len(set(all_gt_labels[1].values()))
plot_data = extract_plot_data(all_pools_dicts, all_gt_labels, all_pred_labels)
#---------------------settings-----------------
# Plot the subfigures
interval = 2
rows = 6
cols = 5
assert rows * cols >= len(plot_data) // interval+1
#---------------------settings-----------------

#Step 5: Plot the subplots in one figure
max_num_subplots = len(plot_data) // interval + 1
fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 4), sharex=True)

for i in range(max_num_subplots):
    if i == 0:      # jump the first epoch 
        continue
    epoch_number = i * interval - 1 
    row = i // cols
    col = i % cols
    create_subplot(axes[row, col], plot_data[epoch_number], epoch_number)

lines, labels = axes[1, 1].get_legend_handles_labels() 
fig.legend(lines, labels, loc = 'upper right', ncol=2, framealpha=0.5)

fig.subplots_adjust(hspace=0.27, wspace=0.2) # Adding extra space for the non overlapping effect
plt.show()

# %%



