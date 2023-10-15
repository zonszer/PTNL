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
path = '../analyze_result_temp/logits&labels_10.14'

#+++++++=========== figure5, dataset is val -> ce PLL00     #NOTE 记得需要每次手动改一下最后一个.pt name -> _test

#+++++++=========== figure1, dataset is train -> PLL05 confidence_RC+gce_rc_PLL0.5-4.pt
# Step 2: Get list of files confidence_RC+gce_rc_PLL0.5-42
files_no_true = sorted(glob.glob(path+ '/conf-cc_refine_1epoch_PLL0.3-*.pt'),  key=lambda x: int(re.findall('(\d+)', x)[-1]))
files_true = sorted(glob.glob(path+ '/conf-cc_refine_1epoch_PLL1e-30-*.pt'),  key=lambda x: int(re.findall('(\d+)', x)[-1]))
fig_id = 'rc_refine-conf-'
##============ set params:


# Step 3: Load data from files into their respective lists
print(f'files_no_true is {files_no_true} \n')
print(f'files_true is {files_true} \n')
stack_list = []
for file in files_no_true:
    tensor = torch.load(file)    # Load tensor from file
    stack_list.append(tensor)

all_data_no_true = torch.stack(stack_list, dim=1)

stack_list_ = []
for file in files_true:
    tensor = torch.load(file)    # Load tensor from file
    stack_list_.append(tensor)

all_data = torch.stack(stack_list_, dim=1)
all_data_bool = all_data.bool()

all_data_no_true = all_data_no_true.cpu().numpy()
all_data_bool = all_data_bool.cpu().numpy()

print(f'all_data_no_true.shape is {all_data_no_true.shape}')



# %%
##============set params:
dataset_class_names = ['Apply_Eye_Makeup', 'Apply_Lipstick', 'Archery', 'Baby_Crawling', 'Balance_Beam', 'Band_Marching', 'Baseball_Pitch', 'Basketball', 'Basketball_Dunk', 'Bench_Press', 'Biking', 'Billiards', 'Blow_Dry_Hair', 'Blowing_Candles', 'Body_Weight_Squats', 'Bowling', 'Boxing_Punching_Bag', 'Boxing_Speed_Bag', 'Breast_Stroke', 'Brushing_Teeth', 'Clean_And_Jerk', 'Cliff_Diving', 'Cricket_Bowling', 'Cricket_Shot', 'Cutting_In_Kitchen', 'Diving', 'Drumming', 'Fencing', 'Field_Hockey_Penalty', 'Floor_Gymnastics', 'Frisbee_Catch', 'Front_Crawl', 'Golf_Swing', 'Haircut', 'Hammering', 'Hammer_Throw', 'Handstand_Pushups', 'Handstand_Walking', 'Head_Massage', 'High_Jump', 'Horse_Race', 'Horse_Riding', 'Hula_Hoop', 'Ice_Dancing', 'Javelin_Throw', 'Juggling_Balls', 'Jumping_Jack', 'Jump_Rope', 'Kayaking', 'Knitting', 'Long_Jump', 'Lunges', 'Military_Parade', 'Mixing', 'Mopping_Floor', 'Nunchucks', 'Parallel_Bars', 'Pizza_Tossing', 'Playing_Cello', 'Playing_Daf', 'Playing_Dhol', 'Playing_Flute', 'Playing_Guitar', 'Playing_Piano', 'Playing_Sitar', 'Playing_Tabla', 'Playing_Violin', 'Pole_Vault', 'Pommel_Horse', 'Pull_Ups', 'Punch', 'Push_Ups', 'Rafting', 'Rock_Climbing_Indoor', 'Rope_Climbing', 'Rowing', 'Salsa_Spin', 'Shaving_Beard', 'Shotput', 'Skate_Boarding', 'Skiing', 'Skijet', 'Sky_Diving', 'Soccer_Juggling', 'Soccer_Penalty', 'Still_Rings', 'Sumo_Wrestling', 'Surfing', 'Swing', 'Table_Tennis_Shot', 'Tai_Chi', 'Tennis_Swing', 'Throw_Discus', 'Trampoline_Jumping', 'Typing', 'Uneven_Bars', 'Volleyball_Spiking', 'Walking_With_Dog', 'Wall_Pushups', 'Writing_On_Board', 'Yo_Yo']
sub_figure_num = 9
figure_type = 'Still_Rings'      # Yo_Yo Still_Rings Long_Jump
# pos_loop = True
np.random.seed(0)
##============set params:

import matplotlib.pyplot as plt

# prepare the index and data of the shown figures:
type_idx = dataset_class_names.index(figure_type)
dataset_class_names_ = np.array(dataset_class_names)
print(f'type_idx is {type_idx}->{figure_type}')
##===========1. random sample：
# Find the indices of the figures that satisfy the condition
valid_indices = np.where(all_data_bool[:, 0, type_idx])[0]

# Randomly select sub_figure_num indices
if valid_indices.shape[0] >= sub_figure_num:
    selected_indices = np.random.choice(valid_indices, size=sub_figure_num, replace=False)
else:
    selected_indices = np.random.choice(valid_indices, size=valid_indices.shape[0], replace=False)
    sub_figure_num = valid_indices.shape[0]
print(f'valid_indices is {valid_indices}')
print(f'selected_indices is {selected_indices}')
# Select the corresponding figures
data = all_data_no_true[selected_indices]               #after that the data shape is (8, 25, 100)
data_taget_label_idx = all_data_bool[selected_indices]

##===========2. select in pos direction or neg direction samples:
# if pos_loop:
#     data = all_data_no_true[:sub_figure_num]
#     data_taget_label_idx = all_data_bool[:sub_figure_num]
# else:
#     data = all_data_no_true[-sub_figure_num:]
#     data_taget_label_idx = all_data_bool[-sub_figure_num:]


num_figures, num_epochs, num_dims = data.shape
colors = plt.cm.Blues(np.linspace(0, 0.7, num_epochs))
colors[:, 3] = 0.75  # Adjust alpha values

rows = int(np.ceil(np.sqrt(num_figures)))  # number of rows in subplot grid
cols = int(np.ceil(num_figures / rows))    # number of columns in subplot grid

fig = plt.figure(figsize=(20,20))

# iterate over all figures
for i in range(num_figures):
    ax = fig.add_subplot(rows, cols, i+1, projection='3d')

    xpos, ypos = np.meshgrid(np.arange(1, num_dims+1, 1), np.arange(num_epochs))
    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')
    zpos = np.zeros_like(xpos)

    dx = dy = np.ones_like(zpos)+1
    dz = data[i].flatten('F')

    # Specify the color for each bar
    color_seq = np.array(colors)[ypos.astype(int)]
    
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=color_seq)

    # Plot max values
    max_z = np.max(data[i], axis=1)
    ax.plot([0]*num_epochs, np.arange(num_epochs), max_z, color='r')        #plot another fig here

    # Plot true target class variation:
    true_z = data[data_taget_label_idx].reshape(sub_figure_num, -1)[i]
    ax.plot([0]*num_epochs, np.arange(num_epochs), true_z, color='g', linewidth=2, linestyle='--')

    ax.set_xlabel('Target Classes')
    ax.set_ylabel('Epochs')
    ax.set_zlabel('Confidence Values')
    ax.set_title('Example {} of ({}) indexed {}'.format(i+1, figure_type, type_idx))
    ax.set_yticks(np.arange(0, num_epochs, 1))
    ax.set_xticks(np.arange(0, num_dims, 10))

    ax.view_init(27, 77)    # Change viewing angle

plt.tight_layout()
# # plt.show()
# plt.savefig(f'{fig_id}_distri-{figure_type}_{current_time}.svg')


# %%
#---------------------settings-----------------
# Create a new 2D-figure for the max_z conf plot
fig2 = plt.figure(figsize=(10, 5))

fig_idx = 8  #NOTE fill the wanted fig index here(start from 0)
#---------------------settings-----------------

ax2 = fig2.add_subplot(111)
# Plot max_z values for each figure
for i in range(fig_idx, fig_idx+1):                        
    max_z = np.max(data[i], axis=1)
    ax2.plot(np.arange(num_epochs), max_z, label=f'max conf')

# ax2.set_xlabel('Epochs')
# ax2.set_ylabel('Max Confidence Values')
# ax2.set_title('Max Confidence Values for Each Example')
# ax2.legend()

# plt.tight_layout()
# plt.show()

# # %%
# # Create a new figure for the true_z conf plot
# fig2 = plt.figure(figsize=(10, 5))
# ax2 = fig2.add_subplot(111)

# true_z conf plot
for i in range(fig_idx, fig_idx+1):
    true_z = data[data_taget_label_idx].reshape(sub_figure_num, -1)[i]
    ax2.plot(np.arange(num_epochs), true_z, linestyle='--', label='true conf')

ax2.set_xlabel('Epochs')
ax2.set_ylabel('Max Confidence Values')
ax2.set_title('Max conf and true conf for Each fub-figure: '+f'Example {i+1} of ({figure_type}) indexed {i+1}')
ax2.legend()

plt.tight_layout()
plt.show()

# %%
