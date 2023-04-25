import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#matplotlib inline
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import csv
import os
from sklearn.linear_model import LinearRegression
import plotly 
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.formula.api as smf


plt.rcParams['figure.figsize'] = 8, 5
plt.style.use("fivethirtyeight")
pd.options.plotting.backend = "plotly"

# specify the folder path where the CSV files are located
folder_path = 'D:\Ravikiran\Trainings\AI and ML Python\Project\doi_10.7941_D1N33Q__v6\Building_59\Bldg59_clean data'
file1_path = os.path.join(folder_path, 'zone_temp_exterior.csv')
file2_path = os.path.join(folder_path, 'uft_fan_spd.csv')
file3_path = os.path.join(folder_path, 'rtu_fan_spd.csv')
file4_path = os.path.join(folder_path, 'zone_temp_interior.csv')
file5_path = os.path.join(folder_path, 'rtu_sa_t.csv')
file6_path = os.path.join(folder_path, 'rtu_sa_fr.csv')
file7_path = os.path.join(folder_path, 'rtu_oa_fr.csv')

# # read in the two CSV files
# file1_path = os.path.join(folder_path, 'ashp_cw.csv')
# file2_path = os.path.join(folder_path, 'occ.csv')
# file4_path = os.path.join(folder_path, 'rtu_ra_t.csv')
# file7_path = os.path.join(folder_path, 'zone_temp_sp_c.csv')
# file8_path = os.path.join(folder_path, 'zone_temp_sp_h.csv')
# file9_path = os.path.join(folder_path, 'site_weather.csv')
# file11_path = os.path.join(folder_path, 'zone_co2.csv')
# file12_path = os.path.join(folder_path, 'hp_hws_temp.csv')
# file13_path = os.path.join(folder_path, 'rtu_sa_t_sp.csv')
# file14_path = os.path.join(folder_path, 'rtu_ma_t.csv')
# file17_path = os.path.join(folder_path, 'rtu_oa_t.csv')
# file18_path = os.path.join(folder_path, 'rtu_oa_damper.csv')
# file19_path = os.path.join(folder_path, 'rtu_econ_sp.csv')
# file20_path = os.path.join(folder_path, 'rtu_plenum_p.csv')
# file21_path = os.path.join(folder_path, 'rtu_sa_p_sp.csv')
# file22_path = os.path.join(folder_path, 'ashp_meter.csv')
# file23_path = os.path.join(folder_path, 'ashp_hw.csv')
# file25_path = os.path.join(folder_path, 'uft_hw_valve.csv')

df1 = pd.read_csv(file1_path, index_col='date', parse_dates=True)
df2 = pd.read_csv(file2_path, index_col='date', parse_dates=True)
df3 = pd.read_csv(file3_path, index_col='date', parse_dates=True)
df4 = pd.read_csv(file4_path, index_col='date', parse_dates=True)
df5 = pd.read_csv(file5_path, index_col='date', parse_dates=True)
df6 = pd.read_csv(file6_path, index_col='date', parse_dates=True)
df7 = pd.read_csv(file7_path, index_col='date', parse_dates=True)
# df8 = pd.read_csv(file8_path, index_col='date', parse_dates=True)
# df9 = pd.read_csv(file9_path, index_col='date', parse_dates=True)
# df10 = pd.read_csv(file10_path, index_col='date', parse_dates=True)
# df11 = pd.read_csv(file11_path, index_col='date', parse_dates=True)
# df12 = pd.read_csv(file12_path, index_col='date', parse_dates=True)
# df13 = pd.read_csv(file13_path, index_col='date', parse_dates=True)
# df14 = pd.read_csv(file14_path, index_col='date', parse_dates=True)
# df15 = pd.read_csv(file15_path, index_col='date', parse_dates=True)
# df16 = pd.read_csv(file16_path, index_col='date', parse_dates=True)
# df17 = pd.read_csv(file17_path, index_col='date', parse_dates=True)
# df18 = pd.read_csv(file18_path, index_col='date', parse_dates=True)
# df19 = pd.read_csv(file19_path, index_col='date', parse_dates=True)
# df20 = pd.read_csv(file20_path, index_col='date', parse_dates=True)
# df21 = pd.read_csv(file21_path, index_col='date', parse_dates=True)
# df22 = pd.read_csv(file22_path, index_col='date', parse_dates=True)
# df23 = pd.read_csv(file23_path, index_col='date', parse_dates=True)
# df24 = pd.read_csv(file24_path, index_col='date', parse_dates=True)
# df25 = pd.read_csv(file25_path, index_col='date', parse_dates=True)

# Mergining the data
# #dfs = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19, df20, df21, df22, df23, df24, df25]
# #dfs = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19, df20, df21, df22, df23, df24]

# dfs = [df2, df3, df4, df5, df6, df1, df9, df10, df11, df12, df14, df15, df16, df17, df18, df20, df22, df23, df24]

dfs = [df1, df2, df3, df4, df5, df6, df7]
df_merged = dfs[0]
for i in range(1, len(dfs)):
    df_merged = pd.merge_asof(df_merged, dfs[i], left_index=True, right_index=True, direction='nearest')

df_merged_new=df_merged[['zone_018_temp', 'zone_025_temp', 'zone_026_temp', 'zone_045_temp', 'zone_055_temp', 'zone_056_temp', 'zone_061_temp', 'zone_018_fan_spd', 'zone_025_fan_spd', 'zone_026_fan_spd', 'zone_045_fan_spd', 'zone_055_fan_spd', 'zone_056_fan_spd', 'zone_061_fan_spd', 'rtu_003_fltrd_sa_flow_tn', 'rtu_003_sf_vfd_spd_fbk_tn', 'rtu_003_rf_vfd_spd_fbk_tn', 'rtu_003_sa_temp','rtu_003_oa_flow_tn']]

col_names = df_merged_new.columns;
# iterating the columns
for col in df_merged_new.columns:
    print(col)    

# Get Index for Multiple Column Labels/Names
Unnamed_cols = [col for col in df_merged_new.columns if 'Unnamed' in col]

dfUnnamed_cols = df_merged_new.filter(regex='Unnamed')
print(dfUnnamed_cols)

# Removing the columns names "Unnamed"
for i in Unnamed_cols:
    del df_merged_new[i]

fig, axs = plt.subplots(nrows=len(df_merged_new.columns), figsize=(70, 70))
# Iterate over each column and plot the data on a subplot
for i, col in enumerate(df_merged_new.columns):
    plt.figure(figsize=(20,7))
    #sns.lineplot(data=df_merged_rs,x="date",y=col)
    sns.scatterplot(data=df_merged_new,x="date",y=col)
    plt.xticks(rotation=90)
    plt.show()
    
df_merged_new = df_merged_new[~df_merged_new.index.duplicated(keep='first')]
df_merged_rs = df_merged_new.resample('15T').ffill()    
#df_merged_rs=df_merged_new
fig, axs = plt.subplots(nrows=len(df_merged_rs.columns), figsize=(70, 70))
# Iterate over each column and plot the data on a subplot
for i, col in enumerate(df_merged_rs.columns):
    plt.figure(figsize=(20,7))
    #sns.lineplot(data=df_merged_rs,x="date",y=col)
    sns.scatterplot(data=df_merged_rs,x="date",y=col)
    plt.xticks(rotation=90)
    plt.show()
    

# Getting the column names of zone_*_fan_spd
ra_temp_cols = [col for col in df_merged_rs.columns if 'fan_spd' in col]

dfra_temp_cols = df_merged_rs.filter(regex='fan_spd')
print(dfra_temp_cols)

# Removing the columns names "Occ"
for i in ra_temp_cols:
    del df_merged_rs[i] 

# Calculation of average of Fan speeds
cols_to_sum = dfra_temp_cols.columns[ : dfra_temp_cols.shape[1]]    
df_merged_rs['Avg_zone_fan_speed']=(dfra_temp_cols[cols_to_sum].sum(axis=1))/ dfra_temp_cols.shape[1]

# # Combining the occupancies
# OCC_cols = [col for col in df_merged_rs.columns if 'occ' in col]

# dfocc_cols = df_merged_rs.filter(regex='occ')
# print(dfocc_cols)

# # Removing the columns names "Occ"
# for i in OCC_cols:
#     del df_merged_rs[i]
    
# # Summation of Occupations
# df_merged_rs['Occupation'] = dfocc_cols['occ_third_south'] + dfocc_cols['occ_fourth_south']    

# Fill the NaN values with Mean of each signals

for i, col in enumerate(df_merged_rs.columns):
    #print(col)
    mean_value=df_merged_rs[col].mean()
    df_merged_rs[col].fillna(value=mean_value, inplace=True)
    df_merged_rs[col].isnull().sum()

#result = df_merged_rs.fillna(df_merged_rs.mean())
# #df_merged_rs1 = df_merged_rs.fillna(0)

df_merged_rs.dropna() 

# # Getting the column names of rtu_*_sa_temp
# sa_temp_cols = [col for col in df_merged_rs.columns if 'sa_temp' in col]

# dfsa_temp_cols = df_merged_rs.filter(regex='sa_temp')
# print(dfsa_temp_cols)

# # Removing the columns names "Occ"
# for i in sa_temp_cols:
#     del df_merged_rs[i]


# # #df_new.dropna()    
# # Getting the column names of rtu_*_ra_temp
# ra_temp_cols = [col for col in df_merged_rs.columns if 'ra_temp' in col]

# dfra_temp_cols = df_merged_rs.filter(regex='ra_temp')
# print(dfra_temp_cols)

# # Removing the columns names "Occ"
# for i in ra_temp_cols:
#     del df_merged_rs[i]    

# df_merged_rs['Diff_rtu_001_temp']=dfsa_temp_cols.rtu_001_sa_temp-dfra_temp_cols.rtu_001_ra_temp
# df_merged_rs['Diff_rtu_002_temp']=dfsa_temp_cols.rtu_002_sa_temp-dfra_temp_cols.rtu_002_ra_temp
# df_merged_rs['Diff_rtu_003_temp']=dfsa_temp_cols.rtu_003_sa_temp-dfra_temp_cols.rtu_003_ra_temp
# df_merged_rs['Diff_rtu_004_temp']=dfsa_temp_cols.rtu_004_sa_temp-dfra_temp_cols.rtu_004_ra_temp

# # Getting the column names of rtu_*_ma_temp
# sa_temp_cols = [col for col in df_merged_rs.columns if 'ma_temp' in col]

# dfsa_temp_cols = df_merged_rs.filter(regex='ma_temp')
# print(dfsa_temp_cols)

# # Removing the columns names "Occ"
# for i in sa_temp_cols:
#     del df_merged_rs[i]


# # #df_new.dropna()    
# # Getting the column names of rtu_*_ra_temp
# ra_temp_cols = [col for col in df_merged_rs.columns if 'oa_temp' in col]

# dfra_temp_cols = df_merged_rs.filter(regex='oa_temp')
# print(dfra_temp_cols)

# # Removing the columns names "Occ"
# for i in ra_temp_cols:
#     del df_merged_rs[i]    

# df_merged_rs['Diff_rtu_ma_001_temp']=dfsa_temp_cols.rtu_001_ma_temp-dfra_temp_cols.rtu_001_oa_temp
# df_merged_rs['Diff_rtu_ma_002_temp']=dfsa_temp_cols.rtu_002_ma_temp-dfra_temp_cols.rtu_002_oa_temp
# df_merged_rs['Diff_rtu_ma_003_temp']=dfsa_temp_cols.rtu_003_ma_temp-dfra_temp_cols.rtu_003_oa_temp
# df_merged_rs['Diff_rtu_ma_004_temp']=dfsa_temp_cols.rtu_004_ma_temp-dfra_temp_cols.rtu_004_oa_temp

# # Getting the column names of rtu_*_fltrd_sa_flow_tn
# sa_temp_cols = [col for col in df_merged_rs.columns if 'sa_flow' in col]

# dfsa_temp_cols = df_merged_rs.filter(regex='sa_flow')
# print(dfsa_temp_cols)

# # Removing the columns names "Occ"
# for i in sa_temp_cols:
#     del df_merged_rs[i]


# # #df_new.dropna()    
# # Getting the column names of rtu_*_oa_flow
# ra_temp_cols = [col for col in df_merged_rs.columns if 'oa_flow' in col]

# dfra_temp_cols = df_merged_rs.filter(regex='oa_flow')
# print(dfra_temp_cols)

# # Removing the columns names "Occ"
# for i in ra_temp_cols:
#     del df_merged_rs[i]    

# df_merged_rs['Diff_air_flow_001']=dfsa_temp_cols.rtu_001_fltrd_sa_flow_tn-dfra_temp_cols.rtu_001_oa_flow_tn
# df_merged_rs['Diff_air_flow_002']=dfsa_temp_cols.rtu_002_fltrd_sa_flow_tn-dfra_temp_cols.rtu_002_oa_flow_tn
# df_merged_rs['Diff_air_flow_003']=dfsa_temp_cols.rtu_003_fltrd_sa_flow_tn-dfra_temp_cols.rtu_003_oa_flow_tn
# df_merged_rs['Diff_air_flow_004']=dfsa_temp_cols.rtu_004_fltrd_sa_flow_tn-dfra_temp_cols.rtu_004_oa_flow_tn

# # Getting the column names of rtu_*_sf_vfd_spd_fbk_tn
# sa_temp_cols = [col for col in df_merged_rs.columns if 'sf_vfd_spd_fbk_tn' in col]

# dfsa_temp_cols = df_merged_rs.filter(regex='sf_vfd_spd_fbk_tn')
# print(dfsa_temp_cols)

# # Removing the columns names "Occ"
# for i in sa_temp_cols:
#     del df_merged_rs[i]


# # #df_new.dropna()    
# # Getting the column names of rtu_*_rf_vfd_spd_fbk_tn
# ra_temp_cols = [col for col in df_merged_rs.columns if 'rf_vfd_spd_fbk_tn' in col]

# dfra_temp_cols = df_merged_rs.filter(regex='rf_vfd_spd_fbk_tn')
# print(dfra_temp_cols)

# # Removing the columns names "Occ"
# for i in ra_temp_cols:
#     del df_merged_rs[i]    

# df_merged_rs['Diff_Fan_spd_001']=dfsa_temp_cols.rtu_001_sf_vfd_spd_fbk_tn-dfra_temp_cols.rtu_001_rf_vfd_spd_fbk_tn
# df_merged_rs['Diff_Fan_spd_002']=dfsa_temp_cols.rtu_002_sf_vfd_spd_fbk_tn-dfra_temp_cols.rtu_002_rf_vfd_spd_fbk_tn
# df_merged_rs['Diff_Fan_spd_003']=dfsa_temp_cols.rtu_003_sf_vfd_spd_fbk_tn-dfra_temp_cols.rtu_003_rf_vfd_spd_fbk_tn
# df_merged_rs['Diff_Fan_spd_004']=dfsa_temp_cols.rtu_004_sf_vfd_spd_fbk_tn-dfra_temp_cols.rtu_004_rf_vfd_spd_fbk_tn

# # Chilled water difference
# df_merged_rs['Diff_Chilled_water_temp']=df_merged_rs.aru_001_cws_temp-df_merged_rs.aru_001_cwr_temp
# del df_merged_rs['aru_001_cws_temp']
# del df_merged_rs['aru_001_cwr_temp']

# # Hot water difference
# df_merged_rs['Diff_Hot_water_temp']=df_merged_rs.aru_001_hws_temp-df_merged_rs.aru_001_hwr_temp
# del df_merged_rs['aru_001_hws_temp']
# del df_merged_rs['aru_001_hwr_temp']


# # Getting the column names of zone_*_fan_spd
# ra_temp_cols = [col for col in df_merged_rs.columns if 'fan_spd' in col]

# dfra_temp_cols = df_merged_rs.filter(regex='fan_spd')
# print(dfra_temp_cols)

# # Removing the columns names "Occ"
# for i in ra_temp_cols:
#     del df_merged_rs[i] 

# # Calculation of average of Fan speeds
# cols_to_sum = dfra_temp_cols.columns[ : dfra_temp_cols.shape[1]]    
# df_merged_rs['Diff_zone_fan_speed']=(dfra_temp_cols[cols_to_sum].sum(axis=1))/ dfra_temp_cols.shape[1]


# # fig, axs = plt.subplots(nrows=len(df_merged_rs.columns), figsize=(70, 70))
# # # Iterate over each column and plot the data on a subplot
# # for i, col in enumerate(df_merged_rs.columns):
# #     plt.figure(figsize=(20,7))
# #     #sns.lineplot(data=df_merged_rs,x="date",y=col)
# #     sns.scatterplot(data=df_merged_rs,x="date",y=col)
# #     plt.xticks(rotation=90)
# #     plt.show()

# # Getting the column names of cerc_templogger
# ra_temp_cols = [col for col in df_merged_rs.columns if 'cerc_templogger' in col]

# dfra_temp_cols = df_merged_rs.filter(regex='cerc_templogger')
# print(dfra_temp_cols)

# # Removing the columns names "Occ"
# for i in ra_temp_cols:
#     del df_merged_rs[i] 
    
# df_merged_rs['Zone1_interior_temp']=(dfra_temp_cols.cerc_templogger_1+dfra_temp_cols.cerc_templogger_2+dfra_temp_cols.cerc_templogger_3+dfra_temp_cols.cerc_templogger_4)/4
# df_merged_rs['Zone2_interior_temp']=(dfra_temp_cols.cerc_templogger_5+dfra_temp_cols.cerc_templogger_6+dfra_temp_cols.cerc_templogger_7+dfra_temp_cols.cerc_templogger_8)/4
# df_merged_rs['Zone3_interior_temp']=(dfra_temp_cols.cerc_templogger_9+dfra_temp_cols.cerc_templogger_10+dfra_temp_cols.cerc_templogger_11+dfra_temp_cols.cerc_templogger_12)/4
# df_merged_rs['Zone4_interior_temp']=(dfra_temp_cols.cerc_templogger_13+dfra_temp_cols.cerc_templogger_14+dfra_temp_cols.cerc_templogger_15+dfra_temp_cols.cerc_templogger_16)/4 


# # Getting the column names of Zone temperature of exterior zone
# ra_temp_cols = [col for col in df_merged_rs.columns if 'zone' in col]

# dfra_temp_cols = df_merged_rs.filter(regex='zone')
# print(dfra_temp_cols)

# # Removing the columns names "Occ"
# for i in ra_temp_cols:
#     del df_merged_rs[i]  
    
    
# ra_temp_cols1 = [col for col in dfra_temp_cols.columns if 'temp' in col]

# dfra_temp_cols1 = dfra_temp_cols.filter(regex='temp')
# print(dfra_temp_cols1)

# # Removing the columns names "Occ"
# for i in ra_temp_cols1:
#     del dfra_temp_cols[i]    
    
# df_merged_rs['Zone1_exterior_temp']=dfra_temp_cols1.iloc[:,-dfra_temp_cols1.shape[1]:-dfra_temp_cols1.shape[1]+13].sum(axis=1)/13
# df_merged_rs['Zone2_exterior_temp']=dfra_temp_cols1.iloc[:,-dfra_temp_cols1.shape[1]+13:-dfra_temp_cols1.shape[1]+26].sum(axis=1)/13
# df_merged_rs['Zone3_exterior_temp']=dfra_temp_cols1.iloc[:,-dfra_temp_cols1.shape[1]+26:-dfra_temp_cols1.shape[1]+39].sum(axis=1)/13
# df_merged_rs['Zone4_exterior_temp']=dfra_temp_cols1.iloc[:,-dfra_temp_cols1.shape[1]+39:].sum(axis=1)/12

# ra_temp_cols1 = [col for col in dfra_temp_cols.columns if 'co2' in col]

# dfra_temp_cols1 = dfra_temp_cols.filter(regex='co2')
# print(dfra_temp_cols1)

# # Removing the columns names "Occ"
# for i in ra_temp_cols1:
#     del dfra_temp_cols[i] 
    
# df_merged_rs['Zone1_CO2']=dfra_temp_cols1.iloc[:,-dfra_temp_cols1.shape[1]:-dfra_temp_cols1.shape[1]+3].sum(axis=1)/3
# df_merged_rs['Zone2_CO2']=dfra_temp_cols1.iloc[:,-dfra_temp_cols1.shape[1]+3:-dfra_temp_cols1.shape[1]+6].sum(axis=1)/3
# df_merged_rs['Zone3_CO2']=dfra_temp_cols1.iloc[:,-dfra_temp_cols1.shape[1]+6:-dfra_temp_cols1.shape[1]+9].sum(axis=1)/3
# df_merged_rs['Zone4_CO2']=dfra_temp_cols1.iloc[:,-dfra_temp_cols1.shape[1]+9:].sum(axis=1)/2    
# df_merged_rs[dfra_temp_cols.columns]=dfra_temp_cols[dfra_temp_cols.columns]

fig, axs = plt.subplots(nrows=len(df_merged_rs.columns), figsize=(70, 70))
# Iterate over each column and plot the data on a subplot
for i, col in enumerate(df_merged_rs.columns):
    plt.figure(figsize=(20,7))
    #sns.lineplot(data=df_merged_rs,x="date",y=col)
    sns.scatterplot(data=df_merged_rs,x="date",y=col)
    plt.xticks(rotation=90)
    plt.show()

# Remove the columns which does not have any variations
stats = df_merged_rs.describe()   
for i, col in enumerate(stats.columns):
    #print(col)
    if stats[col][2]<0.0001:
        del df_merged_rs[col]

# Remove row based on stats (mean - 3* std) for all signals
stats = df_merged_rs.describe()
for i, col in enumerate(stats.columns):
    df_merged_rs.size/44
    df_merged_rs=df_merged_rs.drop(df_merged_rs[df_merged_rs[col]<(stats[col][1]-2*stats[col][2])].index)
    df_merged_rs=df_merged_rs.drop(df_merged_rs[df_merged_rs[col]>(stats[col][1]+2*stats[col][2])].index)
    
fig, axs = plt.subplots(nrows=len(df_merged_rs.columns), figsize=(70, 70))
# Iterate over each column and plot the data on a subplot
for i, col in enumerate(df_merged_rs.columns):
    plt.figure(figsize=(20,7))
    #sns.lineplot(data=df_merged_rs,x="date",y=col)
    sns.scatterplot(data=df_merged_rs,x="date",y=col)
    plt.xticks(rotation=90)
    plt.show()

# # Splitting the data based on Occupation >=30
# df_occ_30_70=df_merged_rs[df_merged_rs["Occupation"]>=30]
# # Splitting the data based on Occupation <30
# df_occ_0_30=df_merged_rs[df_merged_rs["Occupation"]<30]
# # Splitting the data based on Occupation = 0
# df_occ_0=df_occ_0_30[df_occ_0_30["Occupation"]==0]
# df_occ_0_30 = df_occ_0_30.drop(df_occ_0_30[df_occ_0_30.Occupation == 0].index)

# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()    
    
    
# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()  

# ##############################################################################
# ###### No occupation data
# ##############################################################################
# fig, axs = plt.subplots(nrows=len(df_occ_0.columns), figsize=(70, 70))
# # Iterate over each column and plot the data on a subplot
# for i, col in enumerate(df_occ_0.columns):
#     plt.figure(figsize=(20,7))
#     #sns.lineplot(data=df_merged_rs,x="date",y=col)
#     sns.scatterplot(data=df_occ_0,x="date",y=col)
#     plt.xticks(rotation=90)
#     plt.show()

# # Correlation matrix    
# df_occ_0.dataframeName = 'No occupation data'
# nRow, nCol = df_occ_0.shape
# print(f'There are {nRow} rows and {nCol} columns')    
# print(df_occ_0.head(5))

# # Correlation matrix:
# plotCorrelationMatrix(df_occ_0, 8)

# #Scatter and density plots:
# plotScatterMatrix(df_occ_0, nCol, nCol)     
    
    
    
# ##############################################################################    
# ###### occupation data (0,30)
# ##############################################################################
# fig, axs = plt.subplots(nrows=len(df_occ_0_30.columns), figsize=(70, 70))
# # Iterate over each column and plot the data on a subplot
# for i, col in enumerate(df_occ_0_30.columns):
#     plt.figure(figsize=(20,7))
#     #sns.lineplot(data=df_merged_rs,x="date",y=col)
#     sns.scatterplot(data=df_occ_0_30,x="date",y=col)
#     plt.xticks(rotation=90)
#     plt.show()    

# # Correlation matrix  
# df_occ_0_30.dataframeName = 'occupation data (0,30)'
# nRow, nCol = df_occ_0_30.shape
# print(f'There are {nRow} rows and {nCol} columns')    
# print(df_occ_0_30.head(5))

# # Correlation matrix:
# plotCorrelationMatrix(df_occ_0_30, 8)

# #Scatter and density plots:
# plotScatterMatrix(df_occ_0_30, nCol, nCol) 




# ##############################################################################    
# ###### occupation data [30, 70)
# ##############################################################################
# fig, axs = plt.subplots(nrows=len(df_occ_30_70.columns), figsize=(70, 70))
# # Iterate over each column and plot the data on a subplot
# for i, col in enumerate(df_occ_30_70.columns):
#     plt.figure(figsize=(20,7))
#     #sns.lineplot(data=df_merged_rs,x="date",y=col)
#     sns.scatterplot(data=df_occ_30_70,x="date",y=col)
#     plt.xticks(rotation=90)
#     plt.show() 
    
# # Correlation matrix  
# df_occ_30_70.dataframeName = 'occupation data [30, 70)'
# nRow, nCol = df_occ_30_70.shape
# print(f'There are {nRow} rows and {nCol} columns')    
# print(df_occ_30_70.head(5))

# # Correlation matrix:
# plotCorrelationMatrix(df_occ_30_70, 8)

# #Scatter and density plots:
# plotScatterMatrix(df_occ_30_70, nCol, nCol)     


# Correlation matrix    
df_merged_rs.dataframeName = 'SouthWing Data'
nRow, nCol = df_merged_rs.shape
print(f'There are {nRow} rows and {nCol} columns')    
print(df_merged_rs.head(5))

# Correlation matrix:
plotCorrelationMatrix(df_merged_rs, 8)

#Scatter and density plots:
plotScatterMatrix(df_merged_rs, nCol, nCol)  

corre_matx=df_merged_rs.corr()


fig, axs = plt.subplots(nrows=len(df_merged_rs.columns), figsize=(70, 70))
# Iterate over each column and plot the data on a subplot
for i, col in enumerate(corre_matx.columns):
    plt.figure(figsize=(20,7))
    #sns.lineplot(data=df_merged_rs,x="date",y=col)
    sns.barplot(x=corre_matx.index,y=corre_matx[col],data=df2)
    plt.xticks(rotation=90)
    plt.show()
    

# Splitting y valriables based on rtu
y_rtu_cols = [col for col in df_merged_rs.columns if 'rtu' in col]

y_rtu_cols_val = df_merged_rs.filter(regex='rtu')
print(y_rtu_cols_val)

# Removing the columns names "rtu"
for i in y_rtu_cols:
    del df_merged_rs[i]    
    
x_list=df_merged_rs

######################################################################
############# ML Models
#######################################################################

linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(x_list, y_rtu_cols_val['rtu_003_fltrd_sa_flow_tn'])  # perform linear regression
Y_pred = linear_regressor.predict(x_list)  # make predictions

plt.scatter(y_rtu_cols_val['rtu_003_fltrd_sa_flow_tn'],Y_pred)
plt.plot(y_rtu_cols_val['rtu_003_fltrd_sa_flow_tn'], y_rtu_cols_val['rtu_003_fltrd_sa_flow_tn'], color='red')
plt.show()


linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(x_list, y_rtu_cols_val['rtu_003_sf_vfd_spd_fbk_tn'])  # perform linear regression
Y_pred = linear_regressor.predict(x_list)  # make predictions

plt.scatter(y_rtu_cols_val['rtu_003_sf_vfd_spd_fbk_tn'],Y_pred)
plt.plot(y_rtu_cols_val['rtu_003_sf_vfd_spd_fbk_tn'], y_rtu_cols_val['rtu_003_sf_vfd_spd_fbk_tn'], color='red')
plt.show()


linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(x_list, y_rtu_cols_val['rtu_003_rf_vfd_spd_fbk_tn'])  # perform linear regression
Y_pred = linear_regressor.predict(x_list)  # make predictions

plt.scatter(y_rtu_cols_val['rtu_003_rf_vfd_spd_fbk_tn'],Y_pred)
plt.plot(y_rtu_cols_val['rtu_003_rf_vfd_spd_fbk_tn'], y_rtu_cols_val['rtu_003_rf_vfd_spd_fbk_tn'], color='red')
plt.show()


linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(x_list, y_rtu_cols_val['rtu_003_sa_temp'])  # perform linear regression
Y_pred = linear_regressor.predict(x_list)  # make predictions

plt.scatter(y_rtu_cols_val['rtu_003_sa_temp'],Y_pred)
plt.plot(y_rtu_cols_val['rtu_003_sa_temp'], y_rtu_cols_val['rtu_003_sa_temp'], color='red')
plt.show()

linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(x_list, y_rtu_cols_val['rtu_003_oa_flow_tn'])  # perform linear regression
Y_pred = linear_regressor.predict(x_list)  # make predictions

plt.scatter(y_rtu_cols_val['rtu_003_oa_flow_tn'],Y_pred)
plt.plot(y_rtu_cols_val['rtu_003_oa_flow_tn'], y_rtu_cols_val['rtu_003_oa_flow_tn'], color='red')
plt.show()


# linear_regressor = LinearRegression()  # create object for the class
# linear_regressor.fit(y_rtu_cols_val, x_list['zone_018_temp'])  # perform linear regression
# Y_pred = linear_regressor.predict(y_rtu_cols_val)  # make predictions

# plt.scatter(x_list['zone_018_temp'],Y_pred)
# plt.plot(x_list['zone_018_temp'], x_list['zone_018_temp'], color='red')
# plt.show()

# linear_regressor = LinearRegression()  # create object for the class
# linear_regressor.fit(x_list[['zone_018_temp','zone_025_temp']], x_list['zone_061_fan_spd'])  # perform linear regression
# Y_pred = linear_regressor.predict(x_list[['zone_018_temp','zone_025_temp']])  # make predictions

# plt.scatter(x_list['zone_061_fan_spd'],Y_pred)
# plt.plot(x_list['zone_061_fan_spd'], x_list['zone_061_fan_spd'], color='red')
# plt.show()

# res = smf.ols(y_rtu_cols_val['rtu_003_oa_flow_tn'],x_list).fit()
# print(res.summary())



x1list=x_list.to_numpy()
y_rtu_cols_val1a=y_rtu_cols_val['rtu_003_rf_vfd_spd_fbk_tn'].to_numpy()
# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x1list, y_rtu_cols_val1a)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(x1list)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_rtu_cols_val1a, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_rtu_cols_val1a, diabetes_y_pred))
# Plot outputs
plt.scatter(diabetes_y_pred, y_rtu_cols_val1a, color="black")
plt.plot(y_rtu_cols_val1a, y_rtu_cols_val1a, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()