# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 22:05:04 2023

@author: Rajat
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# specify the folder path where the CSV files are located
folder_path = 'J:/BACKUP_cq40/DriveF/RAJAT/DOCUMENTS/IMPORTANT/AI-ML-CCE/Notebooks-datasets/doi_10_7941_D1N33Q__v6/Building_59/Bldg59_clean data/isolated'

# read in the two CSV files
file1_path = os.path.join(folder_path, 'rtu_ra_t.csv')
file2_path = os.path.join(folder_path, 'rtu_sa_t.csv')
file3_path = os.path.join(folder_path, 'rtu_fan_spd.csv')
file4_path = os.path.join(folder_path, 'occ.csv')
file5_path = os.path.join(folder_path, 'zone_temp_interior.csv')
df1 = pd.read_csv(file1_path, index_col='date', parse_dates=True)
df2 = pd.read_csv(file2_path, index_col='date', parse_dates=True)
df3 = pd.read_csv(file3_path, index_col='date', parse_dates=True)
df4 = pd.read_csv(file4_path, index_col='date', parse_dates=True)
df5 = pd.read_csv(file5_path, index_col='date', parse_dates=True)

dfs = [df1, df2, df3, df4, df5]
# df1_rs = df1.resample('15T').ffill()
# df2_rs = df2.resample('15T').ffill()
# df3_rs = df3.resample('15T').ffill()
# df4_rs = df4.resample('15T').ffill()
    
# merge the two dataframes
# merged_df = pd.merge([df1, df2], how='inner', left_index=True, right_index=True)
# merged_df = pd.merge_asof(df1, df2, left_index=True, right_index=True, direction='nearest')
# merged_df = pd.merge([df1, df2, df3, df4], how='inner', left_index=True, right_index=True)
#merged_df = merged_df.dropna()
# merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
# grouped = merged_df.groupby(merged_df.index)
# mean_values = grouped.mean()
# merged_df_rs = merged_df.resample('15T').mean()

df_merged = dfs[0]
for i in range(1, len(dfs)):
    df_merged = pd.merge_asof(df_merged, dfs[i], left_index=True, right_index=True, direction='nearest')

df_merged = df_merged[~df_merged.index.duplicated(keep='first')]
df_merged_rs = df_merged.resample('15T').ffill()

fig, axs = plt.subplots(nrows=len(df_merged_rs.columns), figsize=(70, 70))

# Iterate over each column and plot the data on a subplot
for i, col in enumerate(df_merged_rs.columns):
    axs[i].plot(df_merged_rs[col], label=col)
    axs[i].set_ylabel(col)
    axs[i].legend(loc='upper left')

# overlay of occupants plot
    # ax_twin = axs[-1].twinx()
    # ax_twin.plot(df_merged_rs['occ_third_south'], 'r-', label='occ_third')
    # ax_twin.plot(df_merged_rs['occ_fourth_south'], 'g-', label='occ_fourth')
    # ax_twin.set_ylabel('Temperature')
    # ax_twin.legend(loc='upper right')

# Set the x-axis label for the last subplot
axs[-1].set_xlabel('Datetime')

# Adjust the layout of the subplots
# plt.tight_layout()
plt.subplots_adjust(hspace=0.5)

# Show the plot
plt.show()




# # plot the data using matplotlib
# plt.plot(merged_df['x'], merged_df['y'])
# plt.xlabel('X Label')
# plt.ylabel('Y Label')
# plt.title('Title')
# plt.show()



# # plot the data using matplotlib
# plt.plot(merged_df['occupancy'], merged_df['fan_speed'], 'o')
# plt.xlabel('Occupancy')
# plt.ylabel('Fan Speed')
# plt.title('Occupancy vs Fan Speed')
# plt.show()

