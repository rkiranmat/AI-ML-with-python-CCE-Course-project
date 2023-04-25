import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#matplotlib inline
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import csv
import os

import plotly 
plt.rcParams['figure.figsize'] = 8, 5
plt.style.use("fivethirtyeight")
pd.options.plotting.backend = "plotly"

# specify the folder path where the CSV files are located
folder_path = 'D:\Ravikiran\Trainings\AI and ML Python\Project\doi_10.7941_D1N33Q__v6\Building_59\Bldg59_clean data'

# read in the two CSV files
file1_path = os.path.join(folder_path, 'ashp_cw.csv')
file2_path = os.path.join(folder_path, 'occ.csv')
file3_path = os.path.join(folder_path, 'rtu_fan_spd.csv')
file4_path = os.path.join(folder_path, 'rtu_ra_t.csv')
file5_path = os.path.join(folder_path, 'rtu_sa_t.csv')
file6_path = os.path.join(folder_path, 'zone_temp_interior.csv')
file7_path = os.path.join(folder_path, 'zone_temp_sp_c.csv')
file8_path = os.path.join(folder_path, 'zone_temp_sp_h.csv')
file9_path = os.path.join(folder_path, 'site_weather.csv')
file10_path = os.path.join(folder_path, 'zone_temp_exterior.csv')
file11_path = os.path.join(folder_path, 'zone_co2.csv')
file12_path = os.path.join(folder_path, 'hp_hws_temp.csv')
file13_path = os.path.join(folder_path, 'rtu_sa_t_sp.csv')
file14_path = os.path.join(folder_path, 'rtu_ma_t.csv')
file15_path = os.path.join(folder_path, 'rtu_oa_t.csv')
file16_path = os.path.join(folder_path, 'rtu_sa_fr.csv')
file17_path = os.path.join(folder_path, 'rtu_oa_fr.csv')
file18_path = os.path.join(folder_path, 'rtu_oa_damper.csv')
file19_path = os.path.join(folder_path, 'rtu_econ_sp.csv')
file20_path = os.path.join(folder_path, 'rtu_plenum_p.csv')
file21_path = os.path.join(folder_path, 'rtu_sa_p_sp.csv')
file22_path = os.path.join(folder_path, 'ashp_meter.csv')
file23_path = os.path.join(folder_path, 'ashp_hw.csv')
file24_path = os.path.join(folder_path, 'uft_fan_spd.csv')
file25_path = os.path.join(folder_path, 'uft_hw_valve.csv')

df1 = pd.read_csv(file1_path, index_col='date', parse_dates=True)
df2 = pd.read_csv(file2_path, index_col='date', parse_dates=True)
df3 = pd.read_csv(file3_path, index_col='date', parse_dates=True)
df4 = pd.read_csv(file4_path, index_col='date', parse_dates=True)
df5 = pd.read_csv(file5_path, index_col='date', parse_dates=True)
df6 = pd.read_csv(file6_path, index_col='date', parse_dates=True)
df7 = pd.read_csv(file7_path, index_col='date', parse_dates=True)
df8 = pd.read_csv(file8_path, index_col='date', parse_dates=True)
df9 = pd.read_csv(file9_path, index_col='date', parse_dates=True)
df10 = pd.read_csv(file10_path, index_col='date', parse_dates=True)
df11 = pd.read_csv(file11_path, index_col='date', parse_dates=True)
df12 = pd.read_csv(file12_path, index_col='date', parse_dates=True)
df13 = pd.read_csv(file13_path, index_col='date', parse_dates=True)
df14 = pd.read_csv(file14_path, index_col='date', parse_dates=True)
df15 = pd.read_csv(file15_path, index_col='date', parse_dates=True)
df16 = pd.read_csv(file16_path, index_col='date', parse_dates=True)
df17 = pd.read_csv(file17_path, index_col='date', parse_dates=True)
df18 = pd.read_csv(file18_path, index_col='date', parse_dates=True)
df19 = pd.read_csv(file19_path, index_col='date', parse_dates=True)
df20 = pd.read_csv(file20_path, index_col='date', parse_dates=True)
df21 = pd.read_csv(file21_path, index_col='date', parse_dates=True)
df22 = pd.read_csv(file22_path, index_col='date', parse_dates=True)
df23 = pd.read_csv(file23_path, index_col='date', parse_dates=True)
df24 = pd.read_csv(file24_path, index_col='date', parse_dates=True)
df25 = pd.read_csv(file25_path, index_col='date', parse_dates=True)

# Mergining the data
#dfs = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19, df20, df21, df22, df23, df24, df25]
#dfs = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19, df20, df21, df22, df23, df24]

dfs = [df2, df3, df4, df5, df6, df1, df9, df10, df11, df12, df14, df15, df16, df17, df18, df20, df22, df23, df24]
df_merged = dfs[0]
for i in range(1, len(dfs)):
    df_merged = pd.merge_asof(df_merged, dfs[i], left_index=True, right_index=True, direction='nearest')
    
col_names = df_merged.columns;
# iterating the columns
for col in df_merged.columns:
    print(col)    

# Get Index for Multiple Column Labels/Names
Unnamed_cols = [col for col in df_merged.columns if 'Unnamed' in col]

dfUnnamed_cols = df_merged.filter(regex='Unnamed')
print(dfUnnamed_cols)

# Removing the columns names "Unnamed"
for i in Unnamed_cols:
    del df_merged[i]

    
df_merged = df_merged[~df_merged.index.duplicated(keep='first')]
df_merged_rs = df_merged.resample('15T').ffill()    

fig, axs = plt.subplots(nrows=len(df_merged_rs.columns), figsize=(70, 70))
# Iterate over each column and plot the data on a subplot
for i, col in enumerate(df_merged_rs.columns):
    plt.figure(figsize=(20,7))
    sns.lineplot(data=df_merged_rs,x="date",y=col,marker='o')
    plt.xticks(rotation=90)
    plt.show()
    
# # Combining the occupancies
# OCC_cols = [col for col in df_merged.columns if 'occ' in col]

# dfocc_cols = df_merged.filter(regex='occ')
# print(dfocc_cols)

# # Removing the columns names "Unnamed"
# for i in OCC_cols:
#     del df_merged[i]    