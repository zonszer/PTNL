# In[1]:
import numpy as np
import re

def read_file(file_path):
    label2classname = {}
    pred_is_right = []
    image_logits = []
    image_labels = []

    with open(file_path, 'r') as f:
        for line in f:
            # Split the line into components
            components = line.strip().split(',')

            # Parse each component to the correct data type
            image_index = int(components[0])
            classname = str(components[1])
            image_label = int(components[2])
            pred = int(components[3])
            init = ''
            for x in components[4:-1]:
                init = init + x + ','
            init = init + components[-1]

            # Store the data
            label2classname[image_label] = classname
            pred_is_right.append(bool(pred))
            image_logits.append(eval(init))
            image_labels.append(image_label)

    # Convert lists to numpy arrays for better performance in numerical computations
    pred_is_right = np.array(pred_is_right)
    image_logits = np.array(image_logits)
    image_labels = np.array(image_labels)
    return label2classname, pred_is_right, image_logits, image_labels

def get_classes_acc(path) -> np.ndarray:
    """
    Read a file and extract accuracy values.
    Args:
        path (str): The path to the file.
    Returns:
        np.ndarray: An array of accuracy values.
    """
    # path = 'zero-shot_testdata_' + dataset + '.txt'
    acc_values = []
    with open(path, 'r') as f:
        lines = f.readlines()

    # For each line, find the accuracy value and append it to the list
    for line in lines:
        match = re.search(r'acc: (\d+\.\d+)%', line)
        if match:
            acc_value = float(match.group(1))
            acc_values.append(acc_value)
    acc_array = np.array(acc_values)
    return acc_array

file_path = 'per_image_results_test_37.txt'
acc_file_path = 'per_class_results_test_37.txt'
acc_array = get_classes_acc(acc_file_path)
acc_dict = {i: acc_array[i] for i in range(len(acc_array))}
label2classname, pred_is_right, image_logits, image_labels = read_file(file_path)

# %%

import matplotlib.pyplot as plt
import torch
import math
import matplotlib.ticker as ticker
from collections import Counter
import numpy as np

def plot_logitsDistri(output_teacher_batch, labels_batch, do_softmax=True, max_num_plots=32):
    num_plots = min(len(output_teacher_batch), max_num_plots)  # Limit the number of plots
    num_cols = 6
    num_rows = math.ceil(num_plots / num_cols)  # Calculate the number of rows needed
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(35, 15))

    if do_softmax:
        output_teacher_batch = torch.nn.functional.softmax(torch.tensor(output_teacher_batch) / 0.1, dim=1)
    if isinstance(output_teacher_batch, torch.Tensor):
        output_teacher_batch = output_teacher_batch.numpy()
    if isinstance(labels_batch, torch.Tensor):
        labels_batch = labels_batch.numpy()
    assert isinstance(labels_batch, np.ndarray), "labels_batch must be numpy.ndarray"
    assert isinstance(output_teacher_batch, np.ndarray), "labels_batch must be numpy.ndarray"

    wrong_pred_list = []
    for i, logits in enumerate(output_teacher_batch[:num_plots-1]):  # Limit the number of plots
        row = i // num_cols
        col = i % num_cols
        max_index = np.argmax(logits)
        if max_index == labels_batch[i]:
            color = 'green' 
        else:
            color = 'blue'
            wrong_pred_list.append(max_index)

        axs[row, col].bar(np.arange(len(logits)), logits, color=color)  # arguments are passed to np.histogram
        axs[row, col].set_title(f"Class: {label2classname[labels_batch[i]]} logits distribution", fontsize=9)
        # axs[row, col].set_xlabel("Logits")
        axs[row, col].set_ylabel("Prob")
        axs[row, col].axvline(x=labels_batch[i], color='red')  # add a vertical line at the position of the label
        axs[row, col].axvline(x=max_index, color='orange')       # add a vertical line at the position of the max
        axs[row, col].text(max_index, np.max(logits), f'max label: {max_index}', ha='center', va='bottom')
        
        # Add grid
        axs[row, col].grid(True)
        
        # Set the number of major and minor ticks on the y-axis
        axs[row, col].yaxis.set_major_locator(ticker.MaxNLocator(10))
        axs[row, col].yaxis.set_minor_locator(ticker.MaxNLocator(50))
        
        # Adjust y-axis limits
        axs[row, col].set_ylim(np.min(logits) - abs(np.min(logits))*(0.01), np.max(logits) + abs(np.max(logits))*(0.01))

    # 2. Calculate the count of wrong predictions
    counter = Counter(wrong_pred_list)
    most_common_classes = counter.most_common(5)
    classes, counts = zip(*most_common_classes)  # unzip the list of tuples
    # Add a subfigure at the end
    bars = axs[-1, -1].bar(classes, counts, color='green')
    for bar, class_, count_ in zip(bars, classes, counts):
        height = bar.get_height()
        axs[-1, -1].text(bar.get_x() + bar.get_width() / 2, height, f'Class: {class_}\nnum: {count_}', ha='center', va='bottom', fontsize=10)
    axs[-1, -1].set_title("Top 5 wrong predictions")
    axs[-1, -1].set_xlabel("Classes")
    axs[-1, -1].set_ylabel("Count")
    # Add grid
    axs[-1, -1].grid(True)
    # Set the number of major and minor ticks on the y-axis
    axs[-1, -1].yaxis.set_major_locator(ticker.MaxNLocator(10))
    axs[-1, -1].yaxis.set_minor_locator(ticker.MaxNLocator(50))
    # Adjust y-axis limits
    axs[-1, -1].set_ylim(0, max(counts) + max(counts)*(0.01))

    # plt.tight_layout()
    plt.show()

# index_cls_worst = np.argsort(acc_array)[:10]
# index_cls_worst = np.argsort(acc_array)[:10]
# index_cls_best = np.argsort(acc_array)[-10:]

# for item_idx in index_cls_worst:
#     print(label2classname[item_idx], f'acc: {acc_array[item_idx]}')

def cal_ACC(logits, labels):
    pred = np.argmax(logits, axis=1)
    acc = np.sum(pred==labels) / len(labels)
    return acc

def cal_ACC_foreach_cls(logits, labels):
    pred = np.argmax(logits, axis=1)
    acc_dict = {}
    for class_idx in range(logits.shape[1]):
        class_indices = (labels == class_idx)
        acc_dict[class_idx] = np.sum(pred[class_indices]==labels[class_indices]) / len(labels[class_indices])
    return acc_dict

image_labels = torch.load('zero_shot_labels_true_dtd.pt').numpy()    # (bs, )
image_logits = torch.load('zero_shot_logits_dtd.pt').numpy()   # (bs, 47)
acc_array = cal_ACC_foreach_cls(image_logits, image_labels)
# overall_acc = cal_ACC(logits, image_labels)

index_cls_worst = np.argsort(list(acc_array.values()))[:10]
index_cls_best = np.argsort(list(acc_array.values()))[-10:]
label2classname = {i: i for i in range(image_logits.shape[1])}
for j, label in enumerate(index_cls_worst):
    if j == 0:
        pass
    else:
        label = input(f'press any labe(num) to continue plot the logits distribution of next class, current deault is "{label}"')
        label = np.array(eval(label))
    idxs = np.where(image_labels==label)
    plot_logitsDistri(image_logits[idxs], label.repeat(len(idxs[0])), do_softmax=True)

# %%
# --------------------------plot the relationship between classes acc and CE loss and pre_num:-----------------
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
dict_as_input_format = False

if dict_as_input_format:
    pred_indices = np.array(list(pred_labels.values()))   # Predicted class labels
    image_labels = np.array(list(gt_labels.values()))    # Ground truth class labels

    # Count total occurrences for each class in pred_labels and gt_labels
    class_counts_pred = Counter(pred_labels.values())        
    class_counts_gt = Counter(gt_labels.values())   

    # Calculate the ACC for each class and store it in acc_dict as {class_idx: acc}
    acc_dict = {}
    for class_idx in range(len(class_counts_gt)):
        acc_dict[class_idx] = class_counts_pred[class_idx] / class_counts_gt[class_idx] if class_counts_gt[class_idx] != 0 else 0
    # Convert keys to class labels
    acc_array = np.array(list(acc_dict.values()))

else:
    # Assuming image_logits and image_labels are torch tensors
    image_logits_, image_labels_ = torch.from_numpy(image_logits), torch.from_numpy(image_labels)
    pred_indices = np.argmax(image_logits, axis=1)  #pred_labels is a dict which like {example_idx: pred class index(labels)}

# Count occurrences for each class
class_counts = Counter(pred_indices)        
class_counts_true = Counter(image_labels)   #gt_labels is a dict which like {example_idx: true class index(labels)}

# Convert keys to class labels
class_counts_dict = {class_idx: count-class_counts_true[class_idx] for class_idx, count in class_counts.items()}
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

# Get predicted class indices
pred_indices = torch.argmax(image_logits_, dim=1)
classes_collect_method, method_name = pred_indices, "pred_indices" #image_labels_ or pred_indices

# Calculate cross entropy loss
loss_fn = nn.CrossEntropyLoss(reduction='none')
ce_values = loss_fn(image_logits_, classes_collect_method).detach().numpy()

# Calculate avg CE and ACC for each class
class_ce_dict = {}
for class_idx in range(image_logits_.shape[1]):
    class_indices = (classes_collect_method == class_idx)
    class_ce_dict[class_idx] = ce_values[class_indices].mean()

_, avg_ce_values = zip(*sorted(class_ce_dict.items()))
avg_ce_values = np.array(avg_ce_values)
avg_ce_values_idx = np.argsort(avg_ce_values)

# Create a single figure with subfigures
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 6))

# Plot the first subfigure
# Class Counts Plot
#NOTE class_counts, acc_array, avg_ce_values are one-to-one correspondence by the class index(0~99) from coresponing dict
# NOTE In all sub figures, all the 3 array idx is by x-axis array, from value small to large (because ordered_colors is from small to large)
sc1 = ax1.scatter(class_counts[class_counts_idx], acc_array[class_counts_idx], c=ordered_colors, cmap=colormap, marker='o')
ax1.set_title('Class Counts vs Class ACC')
ax1.set_xlabel('Class Counts')
ax1.set_ylabel('Class ACC')
ax1.grid(True)
cbar1 = plt.colorbar(sc1, ax=ax1, ticks=np.linspace(0, 1, len(class_counts)), label='Class Index')
cbar1.ax.tick_params(labelsize=7)  # Set the size of the legend text

# Plot the second subfigure using the same colormap and class indices
color_class_idx_order = ordered_colors[np.argsort(class_counts_idx)]
sc2 = ax2.scatter(avg_ce_values[avg_ce_values_idx], acc_array[avg_ce_values_idx], c=color_class_idx_order[avg_ce_values_idx], cmap=colormap, marker='o')
ax2.set_title('Avg CE vs Avg ACC, classes_collect_method: ' + method_name)
ax2.set_xlabel('Avg CE')
ax2.set_ylabel('Avg ACC')
ax2.grid(True)
cbar2 = plt.colorbar(sc2, ax=ax2, ticks=np.linspace(0, 1, len(class_counts)), label='Class Index')
cbar2.ax.tick_params(labelsize=7)  # Set the size of the legend text

# Plot the third subfigure using the same colormap and class indices
# class_idx_order = ordered_colors[np.argsort(class_counts_idx)]
# avg_ce_values[avg_ce_values_idx]
CE_rank = [i for i in range(len(avg_ce_values))]
sc3 = ax3.scatter(CE_rank, class_counts[avg_ce_values_idx], c=color_class_idx_order[avg_ce_values_idx], cmap=colormap, marker='o')
ax3.set_title('Avg CE vs Avg ACC, classes_collect_method: ' + method_name)
ax3.set_xlabel('Avg CE')
ax3.set_ylabel('Class Counts')
ax3.grid(True)
cbar3 = plt.colorbar(sc3, ax=ax3, ticks=np.linspace(0, 1, len(class_counts)), label='Class Index')
cbar3.ax.tick_params(labelsize=7)  # Set the size of the legend text

plt.show()


# %%
