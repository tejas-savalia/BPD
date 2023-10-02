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


# data = pd.read_csv('new_d.csv').drop('Unnamed: 0', axis = 1)



# data = data.sort_values(['ID', 'Session', 'Block'], ascending=[True, True, True]).reset_index().drop('index', axis = 1)
# # data['optimal response taken'] = data['StimulusPresentation1.RESP'] == data['StimulusPresentation1.CRESP']





# # # Models using Traces


# traces = pd.read_csv('non_hierarchical_traces_thin5.csv').drop('Unnamed: 0', axis = 1).reset_index()

# rng = np.random.default_rng()
# new_traces = pd.DataFrame({'trace#': np.arange(3000)})
# #To not shuffle
# # for column in traces.columns:
# #     new_traces[column] = rng.permuted(traces[column].values)
# # traces_melt = new_traces.melt(id_vars='trace#', var_name='parameters', value_name='parameter values')
# traces_melt = traces.melt(id_vars='trace#', var_name='parameters', value_name='parameter values')

# traces_melt = traces_melt[traces_melt.parameters.str.contains('subj')]
# traces_melt.loc[traces_melt.parameters.str.contains('Neutral'), 'Session'] = 'Neutral'
# traces_melt.loc[traces_melt.parameters.str.contains('Stressed'), 'Session'] = 'Stressed'
# traces_melt[['parameters', 'ID']] = traces_melt.parameters.str.split('.', expand=True)
# traces_melt.loc[traces_melt.parameters.str.contains('alpha'), 'parameter values'] = np.exp(traces_melt.loc[traces_melt.parameters.str.contains('alpha'), 'parameter values'])/(1+np.exp(traces_melt.loc[traces_melt.parameters.str.contains('alpha'), 'parameter values']))


# # In[10]:


# for ids in traces_melt.ID.unique():
#     traces_melt.loc[traces_melt['ID'] == ids, 'BPD'] = data.loc[data['ID'] == int(ids), 'BPD#'].unique()[0]
#     traces_melt.loc[traces_melt['ID'] == ids, 'Condition'] = data.loc[data['ID'] == int(ids), 'Condition'].unique()[0]
    
# traces_melt['mean_centered_BPD'] = traces_melt['BPD'] - np.mean(traces_melt['BPD'].values)

# def lm_params(trace_no):
#     parameters = 'pos_alpha_subj'
#     temp_data = traces_melt.loc[((traces_melt['trace#'] ==trace_no) & (traces_melt['parameters'].str.startswith(parameters)))].rename(columns={'parameter values':'param_vals'})
    
#     res = smf.ols(formula='param_vals ~  mean_centered_BPD*Session*C(Condition)', data=temp_data).fit()
#     if trace_no%100 == 0:
#         print(trace_no)
#     return res.params

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

traces = pd.read_csv('model_traces/10k4k5p6thin_indsubj_traces_unity_allpairs.csv').drop('Unnamed: 0', axis = 1)
traces['trace#'] = np.tile(np.arange(1000), 148)
# traces['alpha(Neutral)'] = np.exp(traces['alpha(Neutral)'])/(1+np.exp(traces['alpha(Neutral)']))
# traces['pos_alpha(Neutral)'] = np.exp(traces['pos_alpha(Neutral)'])/(1+np.exp(traces['pos_alpha(Neutral)']))
# traces['alpha(Stressed)'] = np.exp(traces['alpha(Stressed)'])/(1+np.exp(traces['alpha(Stressed)']))
# traces['pos_alpha(Stressed)'] = np.exp(traces['pos_alpha(Stressed)'])/(1+np.exp(traces['pos_alpha(Stressed)']))


traces_melt = traces.melt(id_vars=['subject', 'trace#'], var_name='parameters', value_name='param_vals')
traces_melt['Session'] = 'Stressed'
traces_melt.loc[traces_melt['parameters'].str.contains('Neutral'), 'Session'] = 'Neutral'
for sub in new_data['subject'].unique():
    traces_melt.loc[traces_melt['subject'] == sub, 'BPD'] = new_data.loc[new_data['subject'] == sub, 'BPD#'].unique()[0]
    traces_melt.loc[traces_melt['subject'] == sub, 'Condition'] = new_data.loc[new_data['subject'] == sub, 'Condition'].unique()[0]

traces_melt.loc[traces_melt['Condition'] == 1, 'Condition'] = 'Non_Social'
traces_melt.loc[traces_melt['Condition'] == 2, 'Condition'] = 'Social'
traces_melt['mean_centered_BPD'] = traces_melt['BPD'] - np.mean(traces_melt['BPD'].values)


def fit_lm_beta(trace_no):
    param = 'v'
    temp_df = traces_melt.loc[((traces_melt['trace#'] == trace_no) & (traces_melt['parameters'].str.startswith(param)))]
    res = smf.ols(formula='param_vals ~  mean_centered_BPD*Session*Condition + (1|subject)', data=temp_df).fit()
    if trace_no%100 == 0:
        print(trace_no)
    return res.params



trace_nos = np.arange(1000)
pool = mp.Pool()
beta_results = pd.concat(pool.map(fit_lm_beta, trace_nos), axis = 1)
print(beta_results)
beta = beta_results.reset_index().rename(columns={'index':'coefficient'})
beta.to_csv('LR_coefficients/mean_centered_indsub_beta_randeff_allpairs.csv')

def fit_lm_alpha(trace_no):
    param = 'alpha'
    temp_df = traces_melt.loc[((traces_melt['trace#'] == trace_no) & (traces_melt['parameters'].str.startswith(param)))]
    temp_df['param_vals'] = np.exp(temp_df['param_vals'])/(1 + np.exp(temp_df['param_vals']))
    res = smf.ols(formula='param_vals ~  mean_centered_BPD*Session*Condition + (1|subject)', data=temp_df).fit()
    if trace_no%100 == 0:
        print(trace_no)
    return res.params


trace_nos = np.arange(1000)
pool = mp.Pool()
alpha_loss_results = pd.concat(pool.map(fit_lm_alpha, trace_nos), axis = 1)
print(alpha_loss_results)
alpha_loss = alpha_loss_results.reset_index().rename(columns={'index':'coefficient'})
alpha_loss.to_csv('LR_coefficients/mean_centered_indsub_alpha_loss_transformed_randeff_allpairs.csv')

print('Alpha Loss done')


def fit_lm_pos_alpha(trace_no):
    param = 'pos_alpha'
    temp_df = traces_melt.loc[((traces_melt['trace#'] == trace_no) & (traces_melt['parameters'].str.startswith(param)))]
    temp_df['param_vals'] = np.exp(temp_df['param_vals'])/(1 + np.exp(temp_df['param_vals']))

    res = smf.ols(formula='param_vals ~  mean_centered_BPD*Session*Condition + (1|subject)', data=temp_df).fit()
    if trace_no%100 == 0:
        print(trace_no)
    return res.params



trace_nos = np.arange(1000)
pool = mp.Pool()
alpha_gain_results = pd.concat(pool.map(fit_lm_pos_alpha, trace_nos), axis = 1)
print(alpha_gain_results)
alpha_gain = alpha_gain_results.reset_index().rename(columns={'index':'coefficient'})
alpha_gain.to_csv('LR_coefficients/mean_centered_indsub_alpha_gain_transformed_randeff_allpairs.csv')


# def lm_params(trace_no):
#     parameters = 'alpha_subj'
#     temp_data = traces_melt.loc[((traces_melt['trace#'] ==trace_no) & (traces_melt['parameters'].str.startswith(parameters)))].rename(columns={'parameter values':'param_vals'})
    
#     res = smf.ols(formula='param_vals ~  mean_centered_BPD*Session*C(Condition)', data=temp_data).fit()
#     if trace_no%100 == 0:
#         print(trace_no)
#     return res.params



# trace_nos = np.arange(3000)
# pool = mp.Pool()
# alpha_results = pd.concat(pool.map(lm_params, trace_nos), axis = 1)
# print(alpha_results)
# alpha = alpha_results.reset_index().rename(columns={'index':'coefficient'})
# alpha.to_csv('LR coefficients/mean_centered_shuffled_alpha_l.csv')


# def lm_params(trace_no):
#     parameters = 'v'
#     temp_data = traces_melt.loc[((traces_melt['trace#'] ==trace_no) & (traces_melt['parameters'].str.startswith(parameters)))].rename(columns={'parameter values':'param_vals'})
    
#     res = smf.ols(formula='param_vals ~  mean_centered_BPD*Session*C(Condition)', data=temp_data).fit()
#     if trace_no%100 == 0:
#         print(trace_no)
#     return res.params



# trace_nos = np.arange(3000)
# pool = mp.Pool()
# beta_results = pd.concat(pool.map(lm_params, trace_nos), axis = 1)
# print(beta_results)
# beta = beta_results.reset_index().rename(columns={'index':'coefficient'})
# beta.to_csv('LR coefficients/mean_centered_shuffled_beta.csv')

