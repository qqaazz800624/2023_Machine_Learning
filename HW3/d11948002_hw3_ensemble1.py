#%% Ensemble by voting

import pandas as pd



#%%

#reference: https://github.com/pai4451/ML2021/blob/main/hw3/Ensemble1.ipynb

df1 = pd.read_csv('/home/u/qqaazz800624/2023_Machine_Learning/HW3/outputs/d11948002_hw3_model1.csv', index_col = 0)
df2 = pd.read_csv('/home/u/qqaazz800624/2023_Machine_Learning/HW3/outputs/d11948002_hw3_model2.csv', index_col = 0)
df3 = pd.read_csv('/home/u/qqaazz800624/2023_Machine_Learning/HW3/outputs/d11948002_hw3_model3.csv', index_col = 0)
df4 = pd.read_csv('/home/u/qqaazz800624/2023_Machine_Learning/HW3/outputs/d11948002_hw3_model4.csv', index_col = 0)
df5 = pd.read_csv('/home/u/qqaazz800624/2023_Machine_Learning/HW3/outputs/d11948002_hw3_model5.csv', index_col = 0)
df6 = pd.read_csv('/home/u/qqaazz800624/2023_Machine_Learning/HW3/outputs/d11948002_hw3_model6.csv', index_col = 0)
df7 = pd.read_csv('/home/u/qqaazz800624/2023_Machine_Learning/HW3/outputs/d11948002_hw3_model7.csv', index_col = 0)
df_gradescope = pd.read_csv('/home/u/qqaazz800624/2023_Machine_Learning/HW3/outputs/d11948002_hw3_gradescope.csv', index_col = 0)


df_combine = pd.concat([df1, df2, df3, df4, df5, df6, df7, df_gradescope],axis=1, )
df_combine = df_combine.mode(axis=1).dropna(axis=1)

#%%
df_combine = df_combine.astype('int32')
df_combine.columns = ['Class']
test_len = 3000

def pad4(i):
    return "0"*(4-len(str(i)))+str(i)

df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(test_len)]
df["Category"] = df_combine["Class"]
df.to_csv(f"/home/u/qqaazz800624/2023_Machine_Learning/HW3/outputs/d11948002_hw3_ensemble.csv",index = False)


#%%