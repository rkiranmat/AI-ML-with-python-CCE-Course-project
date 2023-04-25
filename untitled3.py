# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 17:16:53 2023

@author: z004kx9t
"""

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
import statsmodels.api as sm
import statsmodels.formula.api as smf
plt.rcParams['figure.figsize'] = 8, 5
plt.style.use("fivethirtyeight")
pd.options.plotting.backend = "plotly"

# specify the folder path where the CSV files are located
folder_path = 'D:\Ravikiran\Trainings\AI and ML Python\Project\doi_10.7941_D1N33Q__v6\Building_59\Bldg59_clean data'
file1_path = os.path.join(folder_path, 'zone_temp_interior.csv')
file2_path = os.path.join(folder_path, 'ele.csv')
file3_path = os.path.join(folder_path, 'zone_temp_exterior.csv')

df1 = pd.read_csv(file1_path, index_col='date', parse_dates=True)
df2 = pd.read_csv(file2_path, index_col='date', parse_dates=True)
df3 = pd.read_csv(file3_path, index_col='date', parse_dates=True)

dfs = [df1, df2, df3]
df_merged = dfs[0]
for i in range(1, len(dfs)):
    df_merged = pd.merge_asof(df_merged, dfs[i], left_index=True, right_index=True, direction='nearest')


# Get Index for Multiple Column Labels/Names
Unnamed_cols = [col for col in df_merged.columns if 'Unnamed' in col]

dfUnnamed_cols = df_merged.filter(regex='Unnamed')
print(dfUnnamed_cols)

# Removing the columns names "Unnamed"
for i in Unnamed_cols:
    del df_merged[i]

# fig, axs = plt.subplots(nrows=len(df_merged.columns), figsize=(70, 70))
# # Iterate over each column and plot the data on a subplot
# for i, col in enumerate(df_merged.columns):
#     plt.figure(figsize=(20,7))
#     #sns.lineplot(data=df_merged_rs,x="date",y=col)
#     sns.scatterplot(data=df_merged,x="date",y=col)
#     plt.xticks(rotation=90)
#     plt.show()   

df_merged_new = df_merged[~df_merged.index.duplicated(keep='first')]
df_merged_rs = df_merged_new.resample('15T').ffill()    
# #df_merged_rs=df_merged_new
# fig, axs = plt.subplots(nrows=len(df_merged_rs.columns), figsize=(70, 70))
# # Iterate over each column and plot the data on a subplot
# for i, col in enumerate(df_merged_rs.columns):
#     plt.figure(figsize=(20,7))
#     #sns.lineplot(data=df_merged_rs,x="date",y=col)
#     sns.scatterplot(data=df_merged_rs,x="date",y=col)
#     plt.xticks(rotation=90)
#     plt.show() 
    
    
# # Chilled water supply/return temperature, F
# plt.figure(figsize=(20,7))
# #sns.lineplot(data=df_merged_rs,x="date",y="cerc_templogger_1",marker='o',label='cerc_templogger_1')
# sns.lineplot(data=df_merged_rs,x="date",y="cerc_templogger_2",marker='o',label='cerc_templogger_2')
# #sns.lineplot(data=df_merged_rs,x="date",y="cerc_templogger_3",marker='o',label='cerc_templogger_3')
# sns.lineplot(data=df_merged_rs,x="date",y="cerc_templogger_4",marker='o',label='cerc_templogger_4')
# plt.xticks(rotation=90)
# plt.show()    

# plt.figure(figsize=(20,7))
# sns.lineplot(data=df_merged_rs,x="date",y="cerc_templogger_9",marker='o',label='cerc_templogger_9')
# sns.lineplot(data=df_merged_rs,x="date",y="cerc_templogger_10",marker='o',label='cerc_templogger_10')
# sns.lineplot(data=df_merged_rs,x="date",y="cerc_templogger_11",marker='o',label='cerc_templogger_11')
# sns.lineplot(data=df_merged_rs,x="date",y="cerc_templogger_12",marker='o',label='cerc_templogger_12')
# plt.xticks(rotation=90)
# plt.show() 

# Get Index for Multiple Column Labels/Names
Unnamed_cols = [col for col in df_merged_rs.columns if 'cerc' in col]

dfUnnamed_cols = df_merged_rs.filter(regex='cerc')
print(dfUnnamed_cols)

# Removing the columns names "Unnamed"
for i in Unnamed_cols:
    del df_merged_rs[i]

# for i, col in enumerate(dfUnnamed_cols.columns):
#     #plt.figure(figsize=(20,7))
#     plt.hist(dfUnnamed_cols[col], bins=1000)
#     #plt.label='cerc_templogger_10'
#     plt.xticks(rotation=90)
#     #plt.xlim[15, 30]
#     plt.show()     

# Stat calculations
stats = dfUnnamed_cols.describe() 

# # Correlation matrix
# def plotCorrelationMatrix(df, graphWidth):
#     filename = df.dataframeName
#     df = df.dropna('columns') # drop columns with NaN
#     df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
#     if df.shape[1] < 2:
#         print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
#         return
#     corr = df.corr()
#     plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
#     corrMat = plt.matshow(corr, fignum = 1)
#     plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
#     plt.yticks(range(len(corr.columns)), corr.columns)
#     plt.gca().xaxis.tick_bottom()
#     plt.colorbar(corrMat)
#     plt.title(f'Correlation Matrix for {filename}', fontsize=15)
#     plt.show()    
    
# # Scatter and density plots
# def plotScatterMatrix(df, plotSize, textSize):
#     df = df.select_dtypes(include =[np.number]) # keep only numerical columns
#     # Remove rows and columns that would lead to df being singular
#     df = df.dropna('columns')
#     df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
#     columnNames = list(df)
#     if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
#         columnNames = columnNames[:10]
#     df = df[columnNames]
#     ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
#     corrs = df.corr().values
#     for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
#         ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
#     plt.suptitle('Scatter and Density Plot')
#     plt.show()  

# # Correlation matrix    
# dfUnnamed_cols.dataframeName = 'SouthWing Data'
# nRow, nCol = dfUnnamed_cols.shape
# print(f'There are {nRow} rows and {nCol} columns')    
# print(dfUnnamed_cols.head(5))    
# # Correlation matrix:
# plotCorrelationMatrix(df_merged_rs, 8)
# #Scatter and density plots:
# plotScatterMatrix(dfUnnamed_cols, nCol, nCol) 
# dfm=dfUnnamed_cols.mean()
# # dfm_df=pd.DataFrame({'title': dfm.values}, index=dfm.index)
# # ax = sns.barplot(data=dfm_df, x=dfm.index, y='title', hue='card_name')   

# # ax = dfm_df.plot.bar(rot=0)

# dfm.plot.bar()

# dfm.plot(kind="bar", figsize=(20,10))

# plt.show()

# plt.figure(figsize=(20,7))
# sns.lineplot(data=df_merged_rs,x="date",y="hvac_N",label='hvac_N')
# plt.xticks(rotation=90)
# plt.show()

# plt.figure(figsize=(20,7))
# sns.lineplot(data=df_merged_rs,x="date",y="hvac_S",label='hvac_S')
# plt.xticks(rotation=90)
# plt.show()

#df_merged_rs=df_merged_rs.drop(df_merged_rs[df_merged_rs["hvac_S"]>0].index)

# plt.figure(figsize=(20,7))
# plt.hist(df_merged_rs["hvac_S"], bins=1000)
# plt.xticks(rotation=90)
# plt.show()
# plt.figure(figsize=(20,7))
# plt.hist(df_merged_rs["hvac_N"], bins=1000)
# plt.xticks(rotation=90)
# plt.show()


df_merged_rs['South_2ndlevel_inte_temp']=(dfUnnamed_cols.cerc_templogger_1+dfUnnamed_cols.cerc_templogger_2+dfUnnamed_cols.cerc_templogger_3+dfUnnamed_cols.cerc_templogger_4)/4
df_merged_rs['South_groundlevel_inte_temp']=(dfUnnamed_cols.cerc_templogger_9+dfUnnamed_cols.cerc_templogger_10+dfUnnamed_cols.cerc_templogger_11+dfUnnamed_cols.cerc_templogger_12)/4
df_merged_rs['North_2ndlevel_inte_temp']=(dfUnnamed_cols.cerc_templogger_5+dfUnnamed_cols.cerc_templogger_6+dfUnnamed_cols.cerc_templogger_7+dfUnnamed_cols.cerc_templogger_8)/4
df_merged_rs['North_groundlevel_inte_temp']=(dfUnnamed_cols.cerc_templogger_13+dfUnnamed_cols.cerc_templogger_14+dfUnnamed_cols.cerc_templogger_15+dfUnnamed_cols.cerc_templogger_16)/4
df_merged_rs['Avg_South_inter_temp']=(df_merged_rs['South_2ndlevel_inte_temp']+df_merged_rs['South_groundlevel_inte_temp'])/2
df_merged_rs['Avg_North_inter_temp']=(df_merged_rs['North_2ndlevel_inte_temp']+df_merged_rs['North_groundlevel_inte_temp'])/2

# plt.figure(figsize=(20,7))
# sns.lineplot(data=df_merged_rs,x="date",y="South_2ndlevel_inte_temp",label='South_2ndlevel_inte_temp')
# plt.xticks(rotation=90)
# plt.show()
# plt.figure(figsize=(20,7))
# sns.lineplot(data=df_merged_rs,x="date",y="South_groundlevel_inte_temp",label='South_groundlevel_inte_temp')
# plt.xticks(rotation=90)
# plt.show()
# plt.figure(figsize=(20,7))
# sns.lineplot(data=df_merged_rs,x="date",y="North_2ndlevel_inte_temp",label='North_2ndlevel_inte_temp')
# plt.xticks(rotation=90)
# plt.show()
# plt.figure(figsize=(20,7))
# sns.lineplot(data=df_merged_rs,x="date",y="North_groundlevel_inte_temp",label='North_groundlevel_inte_temp')
# plt.xticks(rotation=90)
# plt.show()

# plt.figure(figsize=(20,7))
# sns.lineplot(data=df_merged_rs,x="date",y="Avg_South_inter_temp",label='Avg_South_inter_temp')
# plt.xticks(rotation=90)
# plt.show()
# plt.figure(figsize=(20,7))
# sns.lineplot(data=df_merged_rs,x="date",y="Avg_North_inter_temp",label='Avg_North_inter_temp')
# plt.xticks(rotation=90)
# plt.show()

# Get Index for Multiple Column Labels/Names
Unnamed_cols = [col for col in df_merged_rs.columns if 'zone' in col]

dfUnnamed_cols = df_merged_rs.filter(regex='zone')
print(dfUnnamed_cols)

# Removing the columns names "Unnamed"
for i in Unnamed_cols:
    del df_merged_rs[i]
    
df_merged_rs['SouthWing_3_exter_temp']=(dfUnnamed_cols['zone_018_temp']+dfUnnamed_cols['zone_025_temp']+dfUnnamed_cols['zone_026_temp']+dfUnnamed_cols['zone_045_temp']+dfUnnamed_cols['zone_048_temp']+dfUnnamed_cols['zone_055_temp']+dfUnnamed_cols['zone_056_temp']+dfUnnamed_cols['zone_061_temp'])/8  
df_merged_rs['SouthWing_4_exter_temp']=(dfUnnamed_cols['zone_016_temp']+dfUnnamed_cols['zone_017_temp']+dfUnnamed_cols['zone_021_temp']+dfUnnamed_cols['zone_022_temp']+dfUnnamed_cols['zone_023_temp']+dfUnnamed_cols['zone_024_temp']+dfUnnamed_cols['zone_046_temp']+dfUnnamed_cols['zone_047_temp']+dfUnnamed_cols['zone_051_temp']+dfUnnamed_cols['zone_052_temp']+dfUnnamed_cols['zone_053_temp']+dfUnnamed_cols['zone_054_temp'])/12
df_merged_rs['Avg_SouthWing_exter_temp']=(df_merged_rs['SouthWing_3_exter_temp']+df_merged_rs['SouthWing_4_exter_temp'])/2

# plt.figure(figsize=(20,7))
# sns.lineplot(data=df_merged_rs,x="date",y="SouthWing_3_exter_temp",label='SouthWing_3_exter_temp')
# plt.xticks(rotation=90)
# plt.show()
# plt.figure(figsize=(20,7))
# sns.lineplot(data=df_merged_rs,x="date",y="SouthWing_4_exter_temp",label='SouthWing_4_exter_temp')
# plt.xticks(rotation=90)
# plt.show()
# plt.figure(figsize=(20,7))
# sns.lineplot(data=df_merged_rs,x="date",y="Avg_SouthWing_exter_temp",label='Avg_SouthWing_exter_temp')
# plt.xticks(rotation=90)
# plt.show()

df_merged_rs['NorthWing_1_exter_temp']=(dfUnnamed_cols['zone_036_temp']+dfUnnamed_cols['zone_037_temp']+dfUnnamed_cols['zone_038_temp']+dfUnnamed_cols['zone_040_temp']+dfUnnamed_cols['zone_041_temp']+dfUnnamed_cols['zone_042_temp']+dfUnnamed_cols['zone_064_temp']+dfUnnamed_cols['zone_065_temp']+dfUnnamed_cols['zone_066_temp']+dfUnnamed_cols['zone_067_temp']+dfUnnamed_cols['zone_068_temp']+dfUnnamed_cols['zone_069_temp']+dfUnnamed_cols['zone_070_temp'])/13  
df_merged_rs['NorthWing_2_exter_temp']=(dfUnnamed_cols['zone_019_temp']+dfUnnamed_cols['zone_027_temp']+dfUnnamed_cols['zone_028_temp']+dfUnnamed_cols['zone_030_temp']+dfUnnamed_cols['zone_032_temp']+dfUnnamed_cols['zone_033_temp']+dfUnnamed_cols['zone_035_temp']+dfUnnamed_cols['zone_043_temp']+dfUnnamed_cols['zone_044_temp']+dfUnnamed_cols['zone_049_temp']+dfUnnamed_cols['zone_057_temp']+dfUnnamed_cols['zone_058_temp']+dfUnnamed_cols['zone_059_temp']+dfUnnamed_cols['zone_062_temp']+dfUnnamed_cols['zone_063_temp']+dfUnnamed_cols['zone_071_temp']+dfUnnamed_cols['zone_072_temp'])/17
df_merged_rs['Avg_NorthWing_exter_temp']=(df_merged_rs['NorthWing_1_exter_temp']+df_merged_rs['NorthWing_2_exter_temp'])/2
# plt.figure(figsize=(20,7))
# sns.lineplot(data=df_merged_rs,x="date",y="NorthWing_1_exter_temp",label='NorthWing_1_exter_temp')
# plt.xticks(rotation=90)
# plt.show()
# plt.figure(figsize=(20,7))
# sns.lineplot(data=df_merged_rs,x="date",y="NorthWing_2_exter_temp",label='NorthWing_2_exter_temp')
# plt.xticks(rotation=90)
# plt.show()
# plt.figure(figsize=(20,7))
# sns.lineplot(data=df_merged_rs,x="date",y="Avg_NorthWing_exter_temp",label='Avg_NorthWing_exter_temp')
# plt.xticks(rotation=90)
# plt.show()
for i, col in enumerate(df_merged_rs.columns):
    #print(col)
    mean_value=df_merged_rs[col].mean()
    df_merged_rs[col].fillna(value=mean_value, inplace=True)
    df_merged_rs[col].isnull().sum()

#result = df_merged_rs.fillna(df_merged_rs.mean())
# #df_merged_rs1 = df_merged_rs.fillna(0)

df_merged_rs.dropna() 

# Filter data between two dates
filtered_df = df_merged_rs.loc[(df_merged_rs.index >= '2019-01-01')
                     & (df_merged_rs.index < '2019-05-01')]



ModelML=filtered_df[['hvac_S','Avg_SouthWing_exter_temp','Avg_South_inter_temp']]
# Remove row based on stats (mean - 3* std) for all signals
stats = ModelML.describe()
for i, col in enumerate(stats.columns):
    ModelML=ModelML.drop(ModelML[ModelML[col]<=0].index)
    
    
# # Remove the columns which does not have any variations
# stats = ModelML.describe()   
# for i, col in enumerate(stats.columns):
#     #print(col)
#     if stats[col][2]<0.0001:
#         del ModelML[col]

# # Remove row based on stats (mean - 3* std) for all signals
# stats = ModelML.describe()
# for i, col in enumerate(stats.columns):
#     ModelML=ModelML.drop(ModelML[ModelML[col]<(stats[col][1]-stats[col][2])].index)
#     ModelML=ModelML.drop(ModelML[ModelML[col]>(stats[col][1]+stats[col][2])].index)    

ModelML=ModelML.drop(ModelML[ModelML['Avg_South_inter_temp']>26].index)
ModelML=ModelML.drop(ModelML[ModelML['hvac_S']>35].index)

sns.lineplot(data=ModelML,x="date",y="Avg_SouthWing_exter_temp",label='Avg_SouthWing_exter_temp')
plt.xticks(rotation=90)
plt.show()

sns.lineplot(data=ModelML,x="date",y="Avg_South_inter_temp",label='Avg_South_inter_temp')
plt.xticks(rotation=90)
plt.show()

sns.lineplot(data=ModelML,x="date",y="hvac_S",label='hvac_S')
plt.xticks(rotation=90)
plt.show()

# ######################################################################
# ############# ML Models
# #######################################################################

# Splitting y valriables based on rtu
y1= ModelML['hvac_S'].to_numpy()
x1=ModelML[['Avg_SouthWing_exter_temp','Avg_South_inter_temp']].to_numpy()

res=sm.OLS(y1,x1).fit()
print(res.summary())
print(res.rsquared)
res_constant = sm.OLS(y1, sm.add_constant(x1)).fit()
print(res_constant.summary())


sns.lineplot(x=ModelML.index,y=y1,label='real data')
#sns.lineplot(x=ModelML.index,y=0.0068*x1[:,0]+0.5424*x1[:,1],label='model data2')
sns.lineplot(x=ModelML.index,y=-219.7805+3.3137*x1[:,0]+0*x1[:,1],label='model data1')
sns.lineplot(x=ModelML.index,y=-160.0721+2.1107*x1[:,0]+1.0487*x1[:,1],label='model data2')

plt.xticks(())
plt.yticks(())
#plt.labels('real data','model data')
plt.xticks(rotation=90)
plt.show()

# ######################################################################
# ############# predictions
# #######################################################################






# Filter data between two dates
filtered_df1 = df_merged_rs.loc[(df_merged_rs.index >= '2019-05-01')
                     & (df_merged_rs.index < '2019-09-01')]



ModelML1=filtered_df1[['hvac_S','Avg_SouthWing_exter_temp','Avg_South_inter_temp']]
# Remove row based on stats (mean - 3* std) for all signals
stats = ModelML1.describe()
for i, col in enumerate(stats.columns):
    ModelML1=ModelML1.drop(ModelML1[ModelML1[col]<=0].index)
    
   
# # Remove the columns which does not have any variations
# stats = ModelML.describe()   
# for i, col in enumerate(stats.columns):
#     #print(col)
#     if stats[col][2]<0.0001:
#         del ModelML[col]

# # Remove row based on stats (mean - 3* std) for all signals
# stats = ModelML.describe()
# for i, col in enumerate(stats.columns):
#     ModelML=ModelML.drop(ModelML[ModelML[col]<(stats[col][1]-stats[col][2])].index)
#     ModelML=ModelML.drop(ModelML[ModelML[col]>(stats[col][1]+stats[col][2])].index)    

ModelML1=ModelML1.drop(ModelML1[ModelML1['Avg_South_inter_temp']>26].index)
ModelML1=ModelML1.drop(ModelML1[ModelML1['hvac_S']>35].index)

sns.lineplot(data=ModelML1,x="date",y="Avg_SouthWing_exter_temp",label='Avg_SouthWing_exter_temp')
plt.xticks(rotation=90)
plt.show()

sns.lineplot(data=ModelML1,x="date",y="Avg_South_inter_temp",label='Avg_South_inter_temp')
plt.xticks(rotation=90)
plt.show()

sns.lineplot(data=ModelML1,x="date",y="hvac_S",label='hvac_S')
plt.xticks(rotation=90)
plt.show()    


# Splitting y valriables based on rtu
y1= ModelML1['hvac_S'].to_numpy()
x1=ModelML1[['Avg_SouthWing_exter_temp','Avg_South_inter_temp']].to_numpy()
sns.lineplot(x=ModelML1.index,y=y1,label='real data')
#sns.lineplot(x=ModelML.index,y=0.0068*x1[:,0]+0.5424*x1[:,1],label='model data2')
sns.lineplot(x=ModelML1.index,y=-219.7805+3.3137*x1[:,0]+0*x1[:,1],label='model data1')
sns.lineplot(x=ModelML1.index,y=-160.0721+2.1107*x1[:,0]+1.0487*x1[:,1],label='model data2')

plt.xticks(())
plt.yticks(())
#plt.labels('real data','model data')
plt.xticks(rotation=90)
plt.show()