import pandas as pd
import numpy as np
import hddm
import multiprocessing as mp

data = pd.read_csv('new_d.csv')
data = data[['ID', 'Condition', 'Session', 'BPD#']]
new_data = []
subject = 0
for ids in data.ID.unique():
    if len(data.loc[data['ID'] == ids, 'Session'].unique()) < 2:
        continue
    else:
        temp_data = data.loc[data['ID'] == ids]
        temp_data['subject'] = subject
        new_data.append(temp_data)
        subject = subject + 1
new_data = pd.concat(new_data).drop('ID', axis = 1)

print(new_data.head())
# df_to_fit  = pd.read_csv('df_neutral.csv')
df_neutral  = pd.read_csv('df_neutral.csv')
df_stressed  = pd.read_csv('df_stressed.csv')
df_to_fit = pd.concat([df_neutral, df_stressed])
print(df_to_fit.head())

for i in df_to_fit['subj_idx'].unique():
    # print(new_data.loc[new_data['subject'] == i, 'BPD#'].unique())
    df_to_fit.loc[df_to_fit['subj_idx'] == i, 'BPD'] = new_data.loc[new_data['subject'] == i, 'BPD#'].unique()[0]

df_to_fit.loc[df_to_fit['Condition'] == 1, 'Condition'] = 'Non Social'
df_to_fit.loc[df_to_fit['Condition'] == 2, 'Condition'] = 'Social'
print(df_to_fit)
m_non_hrl = hddm.Hrl(df_to_fit, depends_on = {'v':['Session', 'Condition', 'BPD'], 'alpha':['Session', 'Condition', 'BPD'], 'pos_alpha':['Session', 'Condition', 'BPD']}, p_outlier = 0.05, dual=True)
m_non_hrl.find_starting_values()
m_non_hrl.sample(10000, burn=4000, thin = 6, dbname="traces.db", db="pickle")
subj_traces = m_non_hrl.get_traces()
# subj_traces['chain'] = chain

m_hrl_results_neutral.to_csv('model_traces/10k4k5p6thin_allsubj_traces_unity.csv')

