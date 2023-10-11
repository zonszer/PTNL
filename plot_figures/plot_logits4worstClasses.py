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
label2classname, pred_is_right, image_logits, image_labels = read_file(file_path)

# %%

import matplotlib.pyplot as plt
import torch
import math
import matplotlib.ticker as ticker

def plot_logitsDistri(output_teacher_batch, labels_batch, do_softmax=True):
    num_plots = len(output_teacher_batch)
    num_cols = 8
    num_rows = math.ceil(num_plots / num_cols)  # Calculate the number of rows needed
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(32, 15))

    if do_softmax:
        output_teacher_batch = torch.nn.functional.softmax(torch.tensor(output_teacher_batch), dim=1)
    if isinstance(output_teacher_batch, torch.Tensor):
        output_teacher_batch = output_teacher_batch.numpy()
    if isinstance(labels_batch, torch.Tensor):
        labels_batch = labels_batch.numpy()
    assert isinstance(labels_batch, np.ndarray), "labels_batch must be numpy.ndarray"
    assert isinstance(output_teacher_batch, np.ndarray), "labels_batch must be numpy.ndarray"

    for i, logits in enumerate(output_teacher_batch):
        row = i // 8
        col = i % 8
        max_index = np.argmax(logits)
        color = 'green' if max_index == labels_batch[i] else 'blue'
        axs[row, col].bar(np.arange(len(logits)), logits, color=color)  # arguments are passed to np.histogram
        axs[row, col].set_title(f"{label2classname[labels_batch[i]]} logits distribution")
        axs[row, col].set_xlabel("Logits")
        axs[row, col].set_ylabel("Value")
        axs[row, col].axvline(x=labels_batch[i], color='red')  # add a vertical line at the position of the label
        axs[row, col].axvline(x=max_index, color='purple')       # add a vertical line at the position of the max
        axs[row, col].text(max_index, np.max(logits), f'max label: {max_index}', ha='center', va='bottom')
        
        # Add grid
        axs[row, col].grid(True)
        
        # Set the number of major and minor ticks on the y-axis
        axs[row, col].yaxis.set_major_locator(ticker.MaxNLocator(10))
        axs[row, col].yaxis.set_minor_locator(ticker.MaxNLocator(50))
        
        # Adjust y-axis limits
        axs[row, col].set_ylim(np.min(logits) - abs(np.min(logits))*(0.01), np.max(logits) + abs(np.max(logits))*(0.01))

    plt.tight_layout()
    plt.show()



index_cls_worst = np.argsort(acc_array)[:10]
index_cls_best = np.argsort(acc_array)[-10:]

for item_idx in index_cls_worst:
    print(label2classname[item_idx], f'acc: {acc_array[item_idx]}')

for label in index_cls_worst:
    idxs = np.where(image_labels==label)
    plot_logitsDistri(image_logits[idxs], label.repeat(len(idxs[0])), do_softmax=False)
    input('press any key to continue...')




# %%
