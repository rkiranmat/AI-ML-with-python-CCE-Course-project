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

file1_path = os.path.join(folder_path, 'zone_temp_exterior.csv')
file2_path = os.path.join(folder_path, 'uft_fan_spd.csv')
df1 = pd.read_csv(file1_path, index_col='date', parse_dates=True)
df2 = pd.read_csv(file2_path, index_col='date', parse_dates=True)

dfs = [df1, df2]
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
    
df_merged_new = df_merged[~df_merged.index.duplicated(keep='first')]
df_merged_rs = df_merged_new.resample('15T').ffill() 


for i, col in enumerate(df_merged_rs.columns):
    #print(col)
    mean_value=df_merged_rs[col].mean()
    df_merged_rs[col].fillna(value=mean_value, inplace=True)
    df_merged_rs[col].isnull().sum()

#result = df_merged_rs.fillna(df_merged_rs.mean())
# #df_merged_rs1 = df_merged_rs.fillna(0)

df_merged_rs.dropna() 

# Get Index for Multiple Column Labels/Names
Unnamed_cols = [col for col in df_merged_rs.columns if 'fan' in col]

dfUnnamed_cols = df_merged_rs.filter(regex='fan')
print(dfUnnamed_cols)

# Removing the columns names "Unnamed"
for i in Unnamed_cols:
    del df_merged_rs[i]

DFZTEMP=df_merged_rs[['zone_018_temp', 'zone_025_temp', 'zone_026_temp', 'zone_045_temp', 'zone_055_temp', 'zone_056_temp', 'zone_061_temp', 'zone_016_temp', 'zone_017_temp', 'zone_021_temp', 'zone_022_temp', 'zone_023_temp', 'zone_024_temp', 'zone_046_temp', 'zone_047_temp', 'zone_051_temp', 'zone_052_temp', 'zone_053_temp', 'zone_054_temp']]

DFNEW=[dfUnnamed_cols,DFZTEMP]
DFNEW_merged = DFNEW[0]
for i in range(1, len(DFNEW)):
    DFNEW_merged = pd.merge_asof(DFNEW_merged, DFNEW[i], left_index=True, right_index=True, direction='nearest')
    
    
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
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=5)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=5)
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

# Correlation matrix    
DFNEW_merged.dataframeName = 'SouthWing Data'
nRow, nCol = DFNEW_merged.shape
print(f'There are {nRow} rows and {nCol} columns')    
print(DFNEW_merged.head(5))    
# Correlation matrix:
plotCorrelationMatrix(DFNEW_merged, 8)
#Scatter and density plots:
plotScatterMatrix(DFNEW_merged, nCol, nCol)  

corre_matx=DFNEW_merged.corr()


fig, axs = plt.subplots(nrows=len(df_merged_rs.columns), figsize=(70, 70))
# Iterate over each column and plot the data on a subplot
for i, col in enumerate(corre_matx.columns):
    plt.figure(figsize=(20,7))
    #sns.lineplot(data=df_merged_rs,x="date",y=col)
    sns.barplot(x=corre_matx.index,y=corre_matx[col],data=df2)
    plt.xticks(rotation=90)
    plt.show()   