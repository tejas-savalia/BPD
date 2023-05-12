#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install pymc
# !pip install pyreadstat 
# sav_data = pd.read_spss('sav_data.sav')
# sav_data.to_csv('csv_data.csv', index=False)


# In[1]:


#@title Imports and load google drive, navigate to BPD folder
import pandas as pd
import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf
import multiprocessing as mp


# In[2]:


data = pd.read_csv('work/new_d.csv').drop('Unnamed: 0', axis = 1)



data = data.sort_values(['ID', 'Session', 'Block'], ascending=[True, True, 'True']).reset_index().drop('index', axis = 1)
data['optimal response taken'] = data['StimulusPresentation1.RESP'] == data['StimulusPresentation1.CRESP']





# # Models using Traces


traces = pd.read_csv('work/non_hierarchical_traces.csv').drop('Unnamed: 0', axis = 1).reset_index().rename(columns = {'index':'trace#'})
traces_melt = traces.melt(id_vars='trace#', var_name='parameters', value_name='parameter values')
traces_melt = traces_melt[traces_melt.parameters.str.contains('subj')]
traces_melt.loc[traces_melt.parameters.str.contains('Neutral'), 'Session'] = 'Neutral'
traces_melt.loc[traces_melt.parameters.str.contains('Stressed'), 'Session'] = 'Stressed'
traces_melt[['parameters', 'ID']] = traces_melt.parameters.str.split('.', expand=True)
traces_melt.loc[traces_melt.parameters.str.contains('alpha'), 'parameter values'] = np.exp(traces_melt.loc[traces_melt.parameters.str.contains('alpha'), 'parameter values'])/(1+np.exp(traces_melt.loc[traces_melt.parameters.str.contains('alpha'), 'parameter values']))


# In[10]:


for ids in traces_melt.ID.unique():
    traces_melt.loc[traces_melt['ID'] == ids, 'BPD'] = data.loc[data['ID'] == int(ids), 'BPD#'].unique()[0]
    traces_melt.loc[traces_melt['ID'] == ids, 'Condition'] = data.loc[data['ID'] == int(ids), 'Condition'].unique()[0]
    


def lm_params(trace_no):
    parameters = 'pos_alpha_subj'
    temp_data = traces_melt.loc[((traces_melt['trace#'] ==trace_no) & (traces_melt['parameters'].str.startswith(parameters)))].rename(columns={'parameter values':'param_vals'})
    res = smf.ols(formula='param_vals ~  BPD*Session*C(Condition)', data=temp_data).fit()
    if trace_no%100 == 0:
        print(trace_no)
    return res.params



trace_nos = np.arange(3000)
pool = mp.Pool()
pos_alpha_results = pd.concat(pool.map(lm_params, trace_nos), axis = 1)
print(pos_alpha_results)
pos_alpha = pos_alpha_results.reset_index().rename(columns={'index':'coefficient'})
pos_alpha.to_csv('work/alpha_g.csv')
