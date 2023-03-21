#%% Ensemble by voting

import pandas as pd

#%%
#reference: https://github.com/pai4451/ML2021/blob/main/hw3/Ensemble1.ipynb

df1 = pd.read_csv('/home/u/qqaazz800624/2023_Machine_Learning/HW2/outputs/d11948002_hw2_model1.csv', index_col = 0)
df2 = pd.read_csv('/home/u/qqaazz800624/2023_Machine_Learning/HW2/outputs/d11948002_hw2_model2.csv', index_col = 0)
df3 = pd.read_csv('/home/u/qqaazz800624/2023_Machine_Learning/HW2/outputs/d11948002_hw2_model3.csv', index_col = 0)
df4 = pd.read_csv('/home/u/qqaazz800624/2023_Machine_Learning/HW2/outputs/d11948002_hw2_model4.csv', index_col = 0)
df5 = pd.read_csv('/home/u/qqaazz800624/2023_Machine_Learning/HW2/outputs/d11948002_hw2_model5.csv', index_col = 0)
df6 = pd.read_csv('/home/u/qqaazz800624/2023_Machine_Learning/HW2/outputs/ensemble2.csv', index_col = 0)

df_combine = pd.concat([df1, df2, df3, df4, df5, df6],axis=1, )
df_combine = df_combine.mode(axis=1).dropna(axis=1)
#%%
df_combine = df_combine.astype('int32')
df_combine.columns = ['Class']
df_combine.to_csv('/home/u/qqaazz800624/2023_Machine_Learning/HW2/outputs/ensemble.csv',index=True)

#%%