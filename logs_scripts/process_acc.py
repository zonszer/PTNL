#%%
import re

def extract_info(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
    
    data = {}
    exp_pattern = r'id: data-.*'
    acc_pattern = r'(\d+(\.\d{1,2})?%)$'
    key, value = '', ''

    for line in lines:                  #TOorg: The code is to store the last K v in the .txt file 
        if re.search(exp_pattern, line):
            if key and value:  # only store to dict if key and value are both not empty
                data[key] = value
            key = re.search(exp_pattern, line).group()  # update new key
            value = ''  # reset value
        if re.search(acc_pattern, line):
            value = re.search(acc_pattern, line).group()

    if key and value:  # store the last key-value pair
        data[key] = value

    print('len:', len(data), data)
    return data

def formatting_data(data_dict):
    new_dict = {}
    for key, value in data_dict.items():
        new_key = key.split(" ")[1]
        new_key = new_key.replace('rn50_ep50', 'rn50ep50')
        new_key = new_key.replace('ssoxford_pets', 'ssoxfordpets')
        new_key = new_key.replace('loss-rc_cav', 'loss-rc cav')
        new_key = new_key.replace('loss-rc_rc', 'loss-rc rc')
        new_dict[new_key] = value
    return new_dict

# data_dict = extract_info('log_10-04_14-06-35_ssoxford_pets.txt')    #log_10-04_17-35-26_sscaltech101.txt log_10-04_17-35-17_ssucf101.txt  log_10-04_14-06-35_ssoxford_pets.txt
data_dict_new = extract_info('log_test_rc_rc_T-10.11_ssoxford_pets.txt')    #log_10-04_17-35-26_sscaltech101.txt log_10-04_17-35-17_ssucf101.txt  log_10-04_14-06-35_ssucf101.txt
data_dict_old = extract_info('log_10-04_17-35-17_ssucf101-contain-no-beta.txt')
data_dict_new = formatting_data(data_dict_new)
data_dict_old = formatting_data(data_dict_old)


import pandas as pd
import matplotlib.pyplot as plt

def parse2df(data_dict):
    # Step 1: Parse the hyperparameters from the ids and store them in a list of dictionaries
    data = []
    for id, acc in data_dict.items():
        params = id.split('_')  
        param_dict = {param.split('-')[0]: param.split('-')[1] for param in params}
        param_dict['accuracy'] = float(acc.strip('%'))  # add the accuracy to the dictionary
        data.append(param_dict)

    # Step 2: Convert the list of dictionaries into a DataFrame 
    df = pd.DataFrame(data)
    return df

df_new = parse2df(data_dict_new)
df_old = parse2df(data_dict_old)
# merge two data dicts:
df_old.loc[:, "change"] = 'old'
df_new.loc[:, "change"] = 'new'
# df.loc[:, "UseWeightedbeta"] = 'False'
# df_result = pd.concat([df_t, df])
df = pd.concat([df_new, df_old])
print('len df:', len(df))

# Now you can continue with the rest of your code as before, 
# modifying the variables to match the hyperparameters in your DataFrame
#%%
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
import numpy as np
# Filters:
#----------------------settings----------------------
# Select the rows where beta == 1.5 and bs == 32
change = "(df['change']=='new')"
loss = "(df['loss']!='CE')"
# loss = "(df['loss']=='CE')"
# loss = "(df['loss']=='cc')"
# loss = "(df['loss']=='rc cav')"
# beta = "(df['beta']=='0.0')"
# PLL_ratio = "(df['usePLLTrue']=='0.3')"
# seed = "(~((df['seed']=='3') & (df['loss']=='rc cav') & (df['usePLLTrue']=='0.3')))"
select_condiction =   loss + '&' + change   #PLL_ratio  + '&' +

if select_condiction == 'None':
    selected_rows = df
else:
    selected_rows = df[eval(select_condiction)]   # & (df['loss']=='rc cav') & (df['usePLLTrue']=='0.1')
#----------------------settings----------------------

# Group by Variables
grouped_vars = ["T", "change", "usePLLTrue"]        
compar_var = 'accuracy'
grouped_data = selected_rows.groupby(grouped_vars)[compar_var].mean().reset_index()
# Convert the "usePLLTrue" column to float
x_axis_var = grouped_vars[0]
color_axis_var = grouped_vars[1]
sub_fig_var = grouped_vars[2]
grouped_data[x_axis_var] = pd.Categorical(grouped_data[x_axis_var], categories=sorted(grouped_data[x_axis_var].unique()), ordered=True)     #TOorg: 
grouped_data[color_axis_var] = pd.Categorical(grouped_data[color_axis_var], categories=sorted(grouped_data[color_axis_var].unique()), ordered=True) 

# Plotting
if len(grouped_vars) == 1:
    g = sns.catplot(data = grouped_data, x = x_axis_var, y = compar_var, kind = 'bar')
    ax = g.ax  # Extract the Axes object from the FacetGrid for further customization
elif len(grouped_vars) == 2:
    g = sns.catplot(data = grouped_data, x = x_axis_var, y = compar_var, hue=color_axis_var, kind = 'bar')
    ax = g.ax
else:
    g = sns.FacetGrid(grouped_data, col=sub_fig_var, col_wrap=3, height=4, aspect=1)
    g.map_dataframe(lambda data, color: sns.barplot(x=x_axis_var, y=compar_var, hue=color_axis_var, data=data, palette='viridis'))
    ax = None  # Set to None as there are multiple axes in FacetGrid
    # Add a legend to each subplot
    for axes in g.axes.flat:
        axes.legend(loc='lower right')

# Despine and add a grid
sns.despine(left=True)
if ax:
    ax.grid(True, which='both', color='gray', linewidth=0.5)
    plt.ylim(grouped_data[compar_var].min() - abs(grouped_data[compar_var].min())*(0.01), grouped_data[compar_var].max() + abs(grouped_data[compar_var].max())*(0.01))
                    
    # Highlight the maximum value
    highlight_index = grouped_data[compar_var].idxmax()
    highlight_value = grouped_data.loc[highlight_index, compar_var]
    highlight_element = [x for x in ax.patches if np.abs(x.get_height() - highlight_value) < 1e-10][0]
    highlight_element.set_facecolor('r')

    # Customize the y ticks
    ax.yaxis.set_major_locator(ticker.MaxNLocator(10))  # Set the number of major ticks
    ax.yaxis.set_minor_locator(ticker.MaxNLocator(50))  # Set the number of minor ticks
    ax.legend(loc='lower right')

else:
    for axes in g.axes.flat:
        axes.grid(True, which='both', color='gray', linewidth=0.5)
        axes.set_ylim(grouped_data[compar_var].min() - abs(grouped_data[compar_var].min())*(0.01), grouped_data[compar_var].max() + abs(grouped_data[compar_var].max())*(0.01))
                    
        # Highlight the maximum value
        highlight_index = grouped_data[compar_var].idxmax()
        highlight_value = grouped_data.loc[highlight_index, compar_var]
        for p in axes.patches:
            if np.abs(p.get_height() - highlight_value) < 1e-10:
                p.set_facecolor('r')
                
        # Customize the y ticks
        axes.yaxis.set_major_locator(ticker.MaxNLocator(10))  # Set the number of major ticks
        axes.yaxis.set_minor_locator(ticker.MaxNLocator(50))  # Set the number of minor ticks


# Show the y values on top of the bars
if ax:
    for p in g.ax.patches:
        ax.annotate("{:.3f}".format(p.get_height()), 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='center', 
                       fontsize=8, color='black', rotation=0, 
                       xytext=(0, 10), 
                       textcoords='offset points')
else:
    for axes in g.axes.flat:  # Loop for all axes
        for p in axes.patches:  
            axes.annotate("{:.3f}".format(p.get_height()), 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='center', 
                       fontsize=8, color='black', rotation=0, 
                       xytext=(0, 10), 
                       textcoords='offset points')

# Add a title to the figure (not the axes)
plt.suptitle(f'{compar_var} VS {grouped_vars} with {select_condiction}', y=1.02)

# Adjust the layout to make room for the title and legend
plt.tight_layout()

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