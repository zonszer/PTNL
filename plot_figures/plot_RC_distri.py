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
path = '../analyze_result_temp/logits&labels'

#+++++++=========== figure1, dataset is train
# Step 2: Get list of files
# files_no_true = sorted(glob.glob(path+ '/confidence_RC-*.pt'), key=lambda x: int(re.search('(\d+)', x).group()))
# files_true = sorted(glob.glob(path+ '/confidence_RC-*_true.pt'), key=lambda x: int(re.search('(\d+)', x).group()))

# # Filter the list of files to only include those that end with a number followed by '.pt'
# files_no_true = [f for f in files_no_true if re.search(r'confidence_RC-\d+\.pt$', f)]
# fig_id = 'RC'
#+++++++=========== figure2, -> PLL05-cc, dataset is val
# files_no_true = sorted(glob.glob(path + '/outputs_CC_PLL05-*.pt'), key=lambda x: int(re.findall('(\d+)', x)[-1]))
# files_true = sorted(glob.glob(path + '/labels_CC_PLL05-*.pt'), key=lambda x: int(re.findall('(\d+)', x)[-1]))
# fig_id = 'CC_PLL05'

#+++++++=========== figure3 -> forward_rc_(), dataset is val
# files_no_true = sorted(glob.glob(path + '/outputs_RC&cc_PLL05-*.pt'), key=lambda x: int(re.findall('(\d+)', x)[-1]))
# files_true = sorted(glob.glob(path + '/labels_RC&cc_PLL05-*.pt'), key=lambda x: int(re.findall('(\d+)', x)[-1]))
# fig_id = 'RC+CC_1'

#+++++++=========== figure4 -> forward_rc(+CC) new(), dataset is val
# files_no_true = sorted(glob.glob(path + '/outputs_RC&cc2_PLL05-*.pt'), key=lambda x: int(re.findall('(\d+)', x)[-1]))
# files_true = sorted(glob.glob(path + '/labels_RC&cc2_PLL05-*.pt'), key=lambda x: int(re.findall('(\d+)', x)[-1]))
# files_no_true = files_no_true[1:]
# files_true = files_true[1:]
# fig_id = 'RC+CC_2'

#+++++++=========== figure5, dataset is val -> ce PLL00     #NOTE 记得需要每次手动改一下最后一个.pt name -> _test
# files_no_true = sorted(glob.glob(path + '/outputs_CE_PLL00-*.pt'), key=lambda x: int(re.findall('(\d+)', x)[-1]))
# files_true = sorted(glob.glob(path + '/labels_CE_PLL00-*.pt'), key=lambda x: int(re.findall('(\d+)', x)[-1]))
# fig_id = 'ce_PLL00'

#+++++++=========== figure6, -> PLL05-ce, dataset is val
# files_no_true = sorted(glob.glob(path + '/outputs_CE_PLL0.5-*.pt'), key=lambda x: int(re.findall('(\d+)', x)[-1]))
# files_true = sorted(glob.glob(path + '/labels_CE_PLL0.5-*.pt'), key=lambda x: int(re.findall('(\d+)', x)[-1]))
# fig_id = 'CE_PLL05'

# #+++++++=========== figure7, -> PLL05-rc, dataset is val
# files_no_true = sorted(glob.glob(path + '/outputs_RC_PLL0.5-*.pt'), key=lambda x: int(re.findall('(\d+)', x)[-1]))
# files_true = sorted(glob.glob(path + '/labels_RC_PLL0.5-*.pt'), key=lambda x: int(re.findall('(\d+)', x)[-1]))
# fig_id = 'RC_PLL05-logits'

#+++++++=========== figure8, dataset is train -> PLL05 confidence_RC+gce_rc_PLL0.5-4.pt
# Step 2: Get list of files confidence_RC+gce_rc_PLL0.5-42
files_no_true = sorted(glob.glob(path+ '/confidence_RC+gce_rc_PLL0.5-*.pt'), key=lambda x: int(re.search('(\d+)', x).group()))
files_true = sorted(glob.glob(path+ '/confidence_true-RC+gce_rc_PLL1e-34-*.pt'), key=lambda x: int(re.search('(\d+)', x).group()))
fig_id = 'RC+gce_rc-conf-'

#+++++++=========== figure9, dataset is train -> PLL05 new_rc confidence-RC_RC_PLL0.5-14.pt
# files_no_true = sorted(glob.glob(path+ '/confidence-RC_RC_PLL0.5-*.pt'), key=lambda x: int(re.search('(\d+)', x).group()))
# files_true = sorted(glob.glob(path+ '/confidence_true-RC_RC_PLL1e-32-*.pt'), key=lambda x: int(re.search('(\d+)', x).group()))
# fig_id = 'RC_new-conf-'
# Step 2: Get list of files confidence-RC_RC_PLL0.1-4.pt
# files_no_true = sorted(glob.glob(path+ '/confidence-RC_RC_PLL0.1-*.pt'), key=lambda x: int(re.search('(\d+)', x).group()))
# files_true = sorted(glob.glob(path+ '/confidence_true-RC_RC_PLL1e-32-*.pt'), key=lambda x: int(re.search('(\d+)', x).group()))
# fig_id = 'RC_new-conf-PLL01'

##============set params:
dataset_class_names = ['face', 'leopard', 'motorbike', 'accordion', 'airplane', 'anchor', 'ant', 'barrel', 'bass', 'beaver', 'binocular', 'bonsai', 'brain', 'brontosaurus', 'buddha', 'butterfly', 'camera', 'cannon', 'car_side', 'ceiling_fan', 'cellphone', 'chair', 'chandelier', 'cougar_body', 'cougar_face', 'crab', 'crayfish', 'crocodile', 'crocodile_head', 'cup', 'dalmatian', 'dollar_bill', 'dolphin', 'dragonfly', 'electric_guitar', 'elephant', 'emu', 'euphonium', 'ewer', 'ferry', 'flamingo', 'flamingo_head', 'garfield', 'gerenuk', 'gramophone', 'grand_piano', 'hawksbill', 'headphone', 'hedgehog', 'helicopter', 'ibis', 'inline_skate', 'joshua_tree', 'kangaroo', 'ketch', 'lamp', 'laptop', 'llama', 'lobster', 'lotus', 'mandolin', 'mayfly', 'menorah', 'metronome', 'minaret', 'nautilus', 'octopus', 'okapi', 'pagoda', 'panda', 'pigeon', 'pizza', 'platypus', 'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone', 'schooner', 'scissors', 'scorpion', 'sea_horse', 'snoopy', 'soccer_ball', 'stapler', 'starfish', 'stegosaurus', 'stop_sign', 'strawberry', 'sunflower', 'tick', 'trilobite', 'umbrella', 'watch', 'water_lilly', 'wheelchair', 'wild_cat', 'windsor_chair', 'wrench', 'yin_yang']
sub_figure_num = 9
figure_type = 'anchor'      #anchor wrench
pos_loop = True
np.random.seed(0)
##============set params:

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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# prepare the index and data of the shown figures:
type_idx = dataset_class_names.index(figure_type)
dataset_class_names = np.array(dataset_class_names)
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

# %%
