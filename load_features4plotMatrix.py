
# %%
import torch
import os
from collections import OrderedDict, defaultdict

pt_path = './analysis_results_test/cifar-100/fp0/200_2_16_random_initend'
label2cname_dict = {0: 'chimpanzee', 1: 'trout', 2: 'skunk', 3: 'spider', 4: 'chair', 5: 'tank', 6: 'keyboard', 7: 'man', 8: 'whale', 9: 'lobster', 10: 'house', 11: 'beetle', 12: 'bear', 13: 'shrew', 14: 'bottle', 15: 'cup', 16: 'bus', 17: 'orange', 18: 'sea', 19: 'oak tree', 20: 'bed', 21: 'tulip', 22: 'rabbit', 23: 'skyscraper', 24: 'apple', 25: 'maple tree', 26: 'pine tree', 27: 'snail', 28: 'pear', 29: 'bridge', 30: 'train', 31: 'mountain', 32: 'caterpillar', 33: 'crocodile', 34: 'snake', 35: 'kangaroo', 36: 'dolphin', 37: 'cattle', 38: 'raccoon', 39: 'mushroom', 40: 'hamster', 41: 'bowl', 42: 'lamp', 43: 'rocket', 44: 'pickup truck', 45: 'wolf', 46: 'worm', 47: 'otter', 48: 'sunflower', 49: 'leopard', 50: 'ray', 51: 'lawn mower', 52: 'motorcycle', 53: 'boy', 54: 'fox', 55: 'palm tree', 56: 'cloud', 57: 'dinosaur', 58: 'turtle', 59: 'forest', 60: 'couch', 61: 'poppy', 62: 'rose', 63: 'bee', 64: 'girl', 65: 'clock', 66: 'can', 67: 'table', 68: 'road', 69: 'orchid', 70: 'streetcar', 71: 'squirrel', 72: 'crab', 73: 'butterfly', 74: 'tractor', 75: 'beaver', 76: 'willow tree', 77: 'camel', 78: 'plain', 79: 'mouse', 80: 'elephant', 81: 'flatfish', 82: 'sweet pepper', 83: 'plate', 84: 'television', 85: 'aquarium fish', 86: 'wardrobe', 87: 'seal', 88: 'lizard', 89: 'cockroach', 90: 'porcupine', 91: 'woman', 92: 'possum', 93: 'baby', 94: 'tiger', 95: 'telephone', 96: 'shark', 97: 'lion', 98: 'castle', 99: 'bicycle'}

# %%
image_features_all = torch.load(os.path.join(pt_path, 'test_v_features.pt')).cpu().numpy()
outputs_all = torch.load(os.path.join(pt_path, 'test_logits.pt')).cpu().numpy()
text_features_all = torch.load(os.path.join(pt_path, 'test_l_features.pt')).cpu().numpy()
label_all = torch.load(os.path.join(pt_path, 'test_labels.pt')).cpu().numpy()

print(image_features_all.shape)
print(outputs_all.shape)
print(text_features_all.shape)
print(label_all.shape)
# output is:
# torch.Size([10000, 512])
# torch.Size([10000, 100])
# torch.Size([100, 512])
# torch.Size([10000])

# %%
# Statistical classes preferences: the classes which are prone to misclassification to
# For each class
misclassified_classes = defaultdict(list)
misclassified_classes_names = defaultdict(list)
for i in range(100):
    # Get the image features for the current class
    class_image_features = image_features_all[label_all == i]
    
    # Sample 100 image features from the current class
    sample_image_features = class_image_features[:100]
    
    # Compute cosine similarity
    cos_sim = cosine_similarity(sample_image_features, text_features_all)
    
    for j in range(100):
        classified_class = np.argmax(cos_sim[j])
        if classified_class != i:
            misclassified_classes[i].append(classified_class)
            misclassified_classes_names[label2cname_dict[i]].append(label2cname_dict[classified_class])



# %%
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import seaborn as sns
import pandas as pd
from collections import defaultdict

# Class names
class_names = label2cname_dict

# Initialize an empty matrix to store the average cosine similarity for each class
cos_sim_matrix = np.zeros((100, 100))

# For each class
for i in range(100):
    # Get the image features for the current class
    class_image_features = image_features_all[label_all == i]
    
    # Sample 100 image features from the current class
    sample_image_features = class_image_features[:100]
    
    # Compute cosine similarity
    cos_sim = cosine_similarity(sample_image_features, text_features_all)
    
    # Average over 100 samples
    cos_sim_avg = cos_sim.mean(axis=0)
    
    # Store the average cosine similarity in the matrix
    cos_sim_matrix[i] = cos_sim_avg

# Convert class names to a list
class_names_list = [f'{class_names[i]}-I' for i in range(100)]
class_names_list_T = [f'{class_names[i]}-T' for i in range(100)]

# Create a dataframe
df = pd.DataFrame(cos_sim_matrix, index=class_names_list, columns=class_names_list_T)

# Identify misclassified classes 
misclassified_classes_ = []
for i in range(100):
    if np.argmax(cos_sim_matrix[i]) != i:
        misclassified_classes_.append(i)

# Identify easily misclassified classes 
misclassified_classes_easy = []
misclassified_classes_easy_names = defaultdict(list)
for class_index, misclassified_class_list in misclassified_classes.items():
    misclassified_num_all = 0
    for i in set(misclassified_class_list):
        misclassified_num = misclassified_class_list.count(i)
        misclassified_num_all +=  misclassified_num
        if misclassified_num > 15:
            misclassified_classes_easy.append((class_index, i, misclassified_num))
            misclassified_classes_easy_names[label2cname_dict[class_index]].append({label2cname_dict[i]: misclassified_num})
        misclassified_classes_easy_names[label2cname_dict[class_index]].append({'wrong/all': f'{misclassified_num_all}/100'})

# Print the misclassified classes
for class_name in misclassified_classes_easy_names:
    if misclassified_classes_easy_names[class_name] != []:
        print(f'{class_name}: {misclassified_classes_easy_names[class_name]}')

# Plot the cosine similarity matrix using seaborn
plt.figure(figsize=(20, 20))
sns.heatmap(df, cmap='hot')

# Add markers for misclassified classes
for class_index in misclassified_classes_:
    plt.scatter(class_index, class_index, color='green', s=100, marker='o')

# Add markers for easily misclassified classes
for class_index in misclassified_classes_easy:
    plt.scatter(class_index[0], class_index[1], facecolors='none', edgecolors='white', s=100, linewidth=3)
    # # Your coordinates for bold text
    # x, y = 0.5, 0.5
    # for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
    plt.text(class_index[0], class_index[1], str(class_index[2]), color='blue', ha='center', va='center', weight='bold', size=12)

plt.title('Cosine Similarity Matrix between Image and Text Features')
plt.show()


# %%
