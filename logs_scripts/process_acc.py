#%%
import re

def extract_info(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
    
    data = {}
    exp_pattern = r'id: data-.*'
    acc_pattern = r'(\d+(\.\d{1,2})?%)$'
    key, value = '', ''
    break_patten = r'------'

    for line in lines:                  #TOorg: The code is to store the last K v in the .txt file 
        # if re.search(break_patten, line):
        #     break
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
        new_key = new_key.replace('rn50_ep50', 'rn50ep50')      #NOTE that here amy need to change
        new_key = new_key.replace('rn50_ep100', 'rn50ep100')
        new_key = new_key.replace('rn50_ep200', 'rn50ep200')

        new_key = new_key.replace('ssoxford_pets', 'ssoxfordpets')
        new_key = new_key.replace('loss-rc_cav', 'loss-rc cav')
        new_key = new_key.replace('loss-lw_lw', 'loss-lw lw')
        new_key = new_key.replace('loss-rc_refine', 'loss-rc refine')
        new_key = new_key.replace('loss-cc_refine', 'loss-cc refine')
        new_key = new_key.replace('loss-lw_refine', 'loss-lw refine')
        new_key = new_key.replace('loss-cav_refine', 'loss-cav refine')
        new_key = new_key.replace('loss-cc_rc', 'loss-cc rc')
        new_key = new_key.replace('loss-rc_rc', 'loss-rc rc')
        new_dict[new_key] = value
    return new_dict

#log_10.17-test_cc_refine_ep100_polishOutput_ssucf101.txt --> （zero not in pool+ output use bef_output)
#log_10.16-test_cc_refine_ep100_recursion_ssucf101.txt  --> not zero so much not in pool+ output use bef_output)
#log_10.16-test_cc_refine_ep100_recursion_nozeroconf_ssucf101.txt --> not zero not in pool
#log_10.18-test_cc_refine_ep100_usenotinpool2_ssucf101.txt --> utilize not in pool by origin_label + temp_pred increment
#after 10.20:
#log_10.22-test_rc_refine_ep100_ssucf101.txt --> test rc_reine with weight of unsafe and safe
# log_10.20-test_cc_refine_ep100_refillpool2_ssucf101.txt --> refill the pool only once, but for cc_refine   (best on ssucf so far)
#log_10.23-retest_rc_refine_ep100_1refill_ssucf101.txt --> retest refill the pool only once(but with cav_conf rather than momn), also with weight of unsafe and safe (best on ssucf so far)

#rc_refine:
#log_10.24-retest_rc_refine_ep100_1refill_ssdtd.txt --> test rc_refine on dtd dataset (TopP=1,2,3,4) (old)
#log_10.25-retest_rc_refine_ep100_1refill_ssdtd.txt --> 四个改进，需要与10.24的对比看是否有提升（这个是HACk版本，即pool cap那里有问题的main branch）
#log_10.26-DEBUG_retest_rc_refine_ep100_improRefill&unc__ssdtd.txt -->四个改进，需要与10.24的对比看是否有提升
#log_10.26-DEBUG_retest_rc_refine_ep100_improRefill&unc_1_ssdtd.txt -->三个改进（去掉pool cap的-1），需要与10.24的对比看是否有提升
#log_10.28-test_cc_refine_ep100_ssdtd.txt --> 把10.28 final 版本的修改加到cc_refine上
#log_10.29-test_rc&cav_refine_ep100_ssdtd.txt --> 做的修改见导图
#log_10.30-test_rc&cav_refine_ep100_ssdtd.txt
#log_10.31-test_rc&cav_refine_ep100_ssdtd.txt
#log_10.31-test_rc&cav_refine_ep100_ssucf101.txt --> 
#log_11.1-test_lw_refine_ep100_ssdtd.txt
#log_11.1-test_lw_refine_ep100_ssucf101.txt
#log_11.2-test_lw&rc_refine_ep100_notuseClean_ssdtd.txt -->
# log_11.2-test_lw&rc_refine_ep100_All_ssucf101.txt
#log_11.5-test_refine_ep100_All-0.8_ssucf101.txt
#log_11.6-test_refine_ep100_All-0.6_ssucf101.txt

#cc_refine:
#log_10.24-test_cc_refine_ep100_1refill_ssdtd.txt --> test cc_refine on dtd dataset (TopP=1,2,3,4) (old)
#log_10.26-DEBUG_test_cc_refine_ep100_1refill_improRefill&unc_ssdtd.txt -->四个改进，需要与10.24的对比看是否有提升
#log_10.26-DEBUG_test_cc_refine_ep100_1refill_improRefill&unc_1cap_ssdtd.txt --> 把pool cap改成回old版本（只升不降, no -1），需要与10.24的对比看是否有提升
# log_10.26-DEBUG_test_cc_refine_ep100_HackCap_ssdtd.txt --> 看看cc用hack版本的pool cap是否有提升
#log_10.27-test_rc&cav_refine_ep100_ssdtd.txt --> 做的修改见导图（10.27以后的结果都默认用了last epoch的结果）
#log_10.28-test_rc&cav_refine_ep100_ssdtd.txt --> 做的修改见导图
#log_10.28-test_rc&cav_refine_ep100_1_ssdtd.txt --> 做的修改见导图
#log_10.28-test_rc&cav_refine_ep100_2_ssdtd.txt --> 做的修改见导图
#log_10.28-test_rc&cav_refine_ep100_final_ssdtd.txt --> 做的修改见导图
#log_10.30-test_cc_refine_ep100_1_ssdtd.txt
#log_10.31-test_cc_refine_ep100_reduceLast_ssdtd.txt
#log_10.31-test_cc_refine_ep100_reduceLast_ssucf101.txt

data_dict_new = extract_info('log_11.6-test_refine_ep100_All-0.6_ssucf101.txt')    #log_10-04_17-35-26_sscaltech101.txt log_10-04_17-35-17_ssucf101.txt  log_10-04_14-06-35_ssucf101.txt
data_dict_new_hack = extract_info('log_10.25-retest_rc_refine_ep100_1refill_ssdtd--LastEpoch.txt')      
data_dict_new1 = extract_info('log_10.26-DEBUG_retest_rc_refine_ep100_improRefill&unc__ssdtd--LastEpoch.txt')      
data_dict_old = extract_info('log_10.24-retest_rc_refine_ep100_1refill_ssdtd--LastEpoch.txt')      

data_dict_new = formatting_data(data_dict_new)
data_dict_new1 = formatting_data(data_dict_new1)
data_dict_new_hack = formatting_data(data_dict_new_hack)
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
df_new1 = parse2df(data_dict_new1)
df_new_hack = parse2df(data_dict_new_hack)
df_old = parse2df(data_dict_old)
# merge two data dicts:
df_old.loc[:, "change"] = 'old'
df_new1.loc[:, "change"] = 'new has-1'
df_new_hack.loc[:, "change"] = 'new_hack'
df_new.loc[:, "change"] = 'new'
# df.loc[:, "UseWeightedbeta"] = 'False'
# df_result = pd.concat([df_t, df])
df = pd.concat([df_new, df_new1, df_new_hack, df_old])
print('len df:', len(df))

#%% NOTE: check the lost experiment settings:
import itertools

def generate_all_combinations(df):
    # Get all unique values for each parameter
    unique_values = {col: df[col].unique() for col in df.columns if col != 'accuracy'}
    
    # Generate all combinations
    all_combinations = list(itertools.product(*unique_values.values())) 
    
    return all_combinations

def check_missing_experiments(df):
    all_combinations = generate_all_combinations(df)
    missing_experiments = []
    for comb in all_combinations:
        if list(comb) not in df.drop(columns='accuracy').values.tolist():
            missing_experiments.append(comb)
    missing_df = pd.DataFrame(missing_experiments, columns=df.columns.drop('accuracy'))
    missing_df['accuracy'] = float('NaN')
    return missing_df

def generate_exp_id(row):
    id_parts = []
    for col in row.index:
        if col != 'accuracy' and col != 'change':
            id_parts.append(f"{col}-{row[col]}")
    id_str = "id: " + "_".join(id_parts)
    id_str = id_str.replace('rn50ep50', 'rn50_ep50')
    id_str = id_str.replace('rn50ep100', 'rn50_ep100')
    id_str = id_str.replace('rn50ep200', 'rn50_ep200')
    id_str = id_str.replace('ssoxfordpets', 'ssoxford_pets')
    id_str = id_str.replace('loss-rc cav', 'loss-rc_cav')
    id_str = id_str.replace('loss-lw lw', 'loss-lw_lw')
    id_str = id_str.replace('loss-rc refine', 'loss-rc_refine')
    id_str = id_str.replace('loss-cav refine', 'loss-cav_refine')
    id_str = id_str.replace('loss-cc refine', 'loss-cc_refine')
    id_str = id_str.replace('loss-lw refine', 'loss-lw_refine')
    id_str = id_str.replace('loss-cc rc', 'loss-cc_rc')
    id_str = id_str.replace('loss-rc rc', 'loss-rc_rc')
    return id_str


# Filter the DataFrame
#----------------------settings----------------------
change_ = "(df['change']=='new')"
# loss_ = "(df['loss']=='cc refine')"
seed_ = "(df['seed']=='2')"
PLL_ratio_ = "(df['usePLLTrue']=='0.3')"

select_condiction_ = PLL_ratio_   +'&'+  change_  +'&'+ seed_
#----------------------settings----------------------

filtered_df = df[eval(select_condiction_)]
# Check for missing experiments
missing_experiments_df = check_missing_experiments(filtered_df)
# Check for missing experiments
print('len missing_exp_ids:', len(missing_experiments_df))
missing_exp_ids = missing_experiments_df.apply(generate_exp_id, axis=1).tolist()


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
# loss = "(df['loss']!='CE')"
loss = "(df['loss']=='cav refine')"
# loss = "(df['loss']=='cc refine')"
# loss = "(df['loss']=='rc refine')"
# loss = "(df['loss']=='lw refine')"
# loss = "(df['loss']=='rc rc') "
# loss = "(df['loss']=='rc cav')"
# beta = "(df['beta']=='0.0')"
PLL_ratio = "(df['usePLLTrue']=='0.1')"
init_R = "(df['initR']=='0.5')"

# 1. for test rc_refine: -- > grouped_vars = ["safeF", "halfW", "topP"]  
MAXPOOL	= "(df['MAXPOOL']=='16')"
# topP = "(df['topP']=='2')"

# 2. for test cc_refine: --> grouped_vars =  ["safeF", "cMomn", "topP"]     
# cMomn = "(df['cMomn']=='0.99')"
# topP = "(df['topP']=='2')"
# MAXPOOL	= "(df['MAXPOOL']=='16')"

# halfW = "(df['halfW']=='0.5')"
# safeF = "(df['safeF']=='0.5')"
# topP = "(df['topP']=='2')"
# Iepoch = "(df['Iepoch']=='1') | (df['Iepoch'].isna())" 
# seed = "(~((df['seed']=='3') & (df['loss']=='rc cav') & (df['usePLLTrue']=='0.3')))"
# seed = "(df['seed']=='1')"
select_condiction = PLL_ratio   +'&'+  change  +'&'+ loss  +'&'+ MAXPOOL +'&'+ init_R 

if select_condiction == 'None':
    selected_rows = df
else:
    selected_rows = df[eval(select_condiction)]   # & (df['loss']=='rc cav') & (df['usePLLTrue']=='0.1')
#----------------------settings----------------------

# Group by Variables
grouped_vars = [ "cMomn", "halfW", "topP"]               
compar_var = 'accuracy'
print("len(selected_rows) :", len(selected_rows))
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
    ax.set_xlabel(x_axis_var)  # Set x-axis label
    ax.set_ylabel(compar_var)  # Set y-axis label
    ax.legend(title=color_axis_var, prop={'size': 6})  # Add legend
elif len(grouped_vars) == 2:
    g = sns.catplot(data = grouped_data, x = x_axis_var, y = compar_var, hue=color_axis_var, kind = 'bar')
    ax = g.ax
    ax.set_xlabel(x_axis_var)  # Set x-axis label
    ax.set_ylabel(compar_var)  # Set y-axis label
    ax.legend(title=color_axis_var, prop={'size': 6})  # Add legend
else:
    g = sns.FacetGrid(grouped_data, col=sub_fig_var, col_wrap=4, height=4, aspect=1)
    g.map_dataframe(lambda data, color: sns.barplot(x=x_axis_var, y=compar_var, hue=color_axis_var, data=data, palette='viridis'))
    ax = None  # Set to None as there are multiple axes in FacetGrid
    # Add a legend and axis labels to each subplot
    for axes in g.axes.flat:
        axes.legend(loc='upper left', title=color_axis_var, prop={'size': 6})
        axes.set_xlabel(x_axis_var)  # Set x-axis label
        axes.set_ylabel(compar_var)  # Set y-axis label

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
    ax.legend(loc='lower left')

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
# %%
