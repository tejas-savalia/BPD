import pandas as pd
import numpy as np
import hddm
import multiprocessing as mp


# df_to_fit  = pd.read_csv('df_neutral.csv')
df_neutral  = pd.read_csv('df_neutral.csv')
df_stressed  = pd.read_csv('df_stressed.csv')
df_to_fit = pd.concat([df_neutral, df_stressed])

def fit_nonhrl(subject):
# def fit_nonhrl(chain):
    
    print('Subject: ', subject)
    temp_df = df_to_fit.loc[df_to_fit.subj_idx == subject].reset_index().drop('index', axis = 1)
    # temp_firsthalf_df = temp_df[:int(len(temp_df)/2)]
    temp_df_AB = temp_df.loc[temp_df['split_by'] == 0].reset_index().drop('index', axis = 1)
    m_non_hrl = hddm.Hrl(temp_df_AB, depends_on = {'v':['Session'], 'alpha':['Session'], 'pos_alpha':['Session']}, p_outlier = 0.05, dual=True)
    m_non_hrl.find_starting_values()
    m_non_hrl.sample(10000, burn=4000, thin = 6, dbname="traces.db", db="pickle")
    subj_traces = m_non_hrl.get_traces()
    subj_traces['subject'] = subject
    # subj_traces['chain'] = chain
    return subj_traces

subj_id = np.arange(148)
# chain_id = np.arange(8)
pool = mp.Pool()
m_hrl_results_neutral = pd.concat(pool.map(fit_nonhrl, subj_id))
# m_hrl_results = pd.concat(pool.map(fit_nonhrl, chain_id))
m_hrl_results_neutral.to_csv('model_traces/10k4k5p6thin_indsubj_AB_traces.csv')


def fit_nonhrl_neutral(subject):
# def fit_nonhrl(chain):
    
    print('Subject: ', subject)
    temp_df = df_neutral.loc[df_neutral.subj_idx == subject].reset_index().drop('index', axis = 1)
    m_non_hrl = hddm.Hrl(temp_df, p_outlier = 0.05, dual=True)
    m_non_hrl.find_starting_values()
    m_non_hrl.sample(10000, burn=4000, thin = 6, dbname="traces.db", db="pickle")
    subj_traces = m_non_hrl.get_traces()
    subj_traces['subject'] = subject
    # subj_traces['chain'] = chain
    return subj_traces

# subj_id = np.arange(148)
# # chain_id = np.arange(8)
# pool = mp.Pool()
# m_hrl_results_neutral = pd.concat(pool.map(fit_nonhrl_neutral, subj_id))
# # m_hrl_results = pd.concat(pool.map(fit_nonhrl, chain_id))
# m_hrl_results_neutral.to_csv('model_traces/10k4k5p6thin_indsubj_neutral_traces.csv')


def fit_nonhrl_stressed(subject):
# def fit_nonhrl(chain):
    
    print('Subject: ', subject)
    temp_df = df_stressed.loc[df_stressed.subj_idx == subject].reset_index().drop('index', axis = 1)
    m_non_hrl = hddm.Hrl(temp_df, p_outlier = 0.05, dual=True)
    m_non_hrl.find_starting_values()
    m_non_hrl.sample(10000, burn=4000, thin = 6, dbname="traces.db", db="pickle")
    subj_traces = m_non_hrl.get_traces()
    subj_traces['subject'] = subject
    # subj_traces['chain'] = chain
    return subj_traces

# subj_id = np.arange(148)
# # chain_id = np.arange(8)
# pool = mp.Pool()
# m_hrl_results_stressed = pd.concat(pool.map(fit_nonhrl_stressed, subj_id))
# # m_hrl_results = pd.concat(pool.map(fit_nonhrl, chain_id))
# m_hrl_results_stressed.to_csv('model_traces/10k4k5p6thin_indsubj_stressed_traces.csv')
