#%%
import re

def extract_info(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
    
    data = {}
    exp_pattern = r'id: data-.*'
    acc_pattern = r'(\d+(\.\d{1,2})?%)$'
    key, value = '', ''

    for line in lines:
        if re.search(exp_pattern, line):
            if key and value:  # only store to dict if key and value are both not empty
                data[key] = value
            key = re.search(exp_pattern, line).group()  # update new key
            value = ''  # reset value
        if re.search(acc_pattern, line):
            value = re.search(acc_pattern, line).group()

    if key and value:  # store the last key-value pair
        data[key] = value

    return data

# data_dict = extract_info('log_10-04_14-06-35_ssoxford_pets.txt')    #log_10-04_17-35-26_sscaltech101.txt log_10-04_17-35-17_ssucf101.txt  log_10-04_14-06-35_ssoxford_pets.txt
data_dict = extract_info('log_10-06_01-40-02_cifar-100.txt')    #log_10-04_17-35-26_sscaltech101.txt log_10-04_17-35-17_ssucf101.txt  log_10-04_14-06-35_ssoxford_pets.txt
new_dict = {}
for key, value in data_dict.items():
    new_key = key.split(" ")[1]
    new_key = new_key.replace('rn50_ep50', 'rn50ep50')
    new_key = new_key.replace('ssoxford_pets', 'ssoxfordpets')
    new_key = new_key.replace('loss-rc_cav', 'loss-rc cav')
    new_key = new_key.replace('loss-rc_rc', 'loss-rc rc')
    new_dict[new_key] = value
data_dict = new_dict
print(len(data_dict), data_dict)

import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Parse the hyperparameters from the ids and store them in a list of dictionaries
data = []
for id, acc in data_dict.items():
    params = id.split('_')  
    param_dict = {param.split('-')[0]: param.split('-')[1] for param in params}
    param_dict['accuracy'] = float(acc.strip('%'))  # add the accuracy to the dictionary
    data.append(param_dict)

# Step 2: Convert the list of dictionaries into a DataFrame 
df = pd.DataFrame(data)
# Now you can continue with the rest of your code as before, 
# modifying the variables to match the hyperparameters in your DataFrame
#%%
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#----------------------settings----------------------
# Select the rows where beta == 1.5 and bs == 32
# loss = "(df['loss']!='CE')"
# loss = "(df['loss']=='cc')"
loss = "(df['loss']=='rc cav')"
# beta = "(df['beta']=='0.0')"
PLL_ratio = "(df['usePLLTrue']=='0.3')"
select_condiction = PLL_ratio + '&' + loss                  # + &


if select_condiction == 'None':
    selected_rows = df
else:
    selected_rows = df[eval(select_condiction)]   # & (df['loss']=='rc cav') & (df['usePLLTrue']=='0.1')
# Step 5: Group the DataFrame by the hyperparameter and calculate the mean of the performance metric
grouped_var = 'beta'
# compar_var = 'head_wiseft_0.5'
compar_var = 'accuracy'
#----------------------settings----------------------

grouped_name = selected_rows.groupby(grouped_var)[compar_var].mean()
# grouped_name = df.groupby(grouped_var)[compar_var].mean()

# Step 6: Visualize the results
ax = grouped_name.plot(kind='bar', color='lightblue')
# Highlight the bar with the maximum value
ax.patches[grouped_name.values.argmax()].set_facecolor('r')

# Set the x and y labels
plt.xlabel(f'{grouped_var}')
plt.ylabel(f'{compar_var}')

# Set the title
plt.title(f'{compar_var} VS {grouped_var}: condition:{select_condiction}')

# Add grid
plt.grid(True, which='both', color='gray', linewidth=0.5)

# Customize the y ticks
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(10))  # Set the number of major ticks
plt.gca().yaxis.set_minor_locator(ticker.MaxNLocator(50))  # Set the number of minor ticks

# Set the y limit
plt.ylim(grouped_name.min() - 0.1, grouped_name.max() + 0.1)
# Show the y values on top of the bars
for i, v in enumerate(grouped_name.values):
    ax.text(i, v + 0.01, "{:.3f}".format(v), ha='center')
plt.show()


# %%
#conclusion for rugulizaion: 
# 1. beta= 1.5 > 1.0 > 0.5 > 0.0
# 2. head > head_wiseft_0.5 
# 3. wd: 0.01 > 0.0001 > 0 (for head_wiseft_0.5 it is reversed)
# 4. lr: head: 0.0001 > 0.001 (for head_wiseft_0.5 it is reversed)
# 5. bs:  32 > 8 
# 6. iter: ACC increase, then decrease with iter increasing, best is around 1400


#best params for head is: beta=1.5(larger better), wd=0.01(not so important), lr=0.0001, bs=32(larger better)