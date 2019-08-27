'''
This solution gives Public Score 0.96038
'''

import os
import pickle
import numpy as np
import pandas as pd
import time
from contextlib import contextmanager
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')


PATH_TO_DATA = '../input'
AUTHOR = 'Pavel_Peskov'

SEED = 17
N_JOBS = 4
NUM_TIME_SPLITS = 10    # for time-based cross-validation
SITE_NGRAMS = (1, 5)    # site ngrams for "bag of sites"
MAX_FEATURES = 50000    # max features for "bag of sites"
LOGIT_C = 3  

# report running times
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')
    

def prepare_sparse_features(path_to_train, path_to_test, path_to_site_dict,
                           vectorizer_params):
    times = ['time%s' % i for i in range(1, 11)]
    train_df = pd.read_csv(path_to_train,
                       index_col='session_id', parse_dates=times)
    test_df = pd.read_csv(path_to_test,
                      index_col='session_id', parse_dates=times)

    # Sort the data by time
    train_df = train_df.sort_values(by='time1')
    
    # read site -> id mapping provided by competition organizers 
    with open(path_to_site_dict, 'rb') as f:
        site2id = pickle.load(f)
    # create an inverse id _> site mapping
    id2site = {v:k for (k, v) in site2id.items()}
    # we treat site with id 0 as "unknown"
    id2site[0] = 'unknown'
    
    # Transform data into format which can be fed into TfidfVectorizer
    # This time we prefer to represent sessions with site names, not site ids. 
    # It's less efficient but thus it'll be more convenient to interpret model weights.
    sites = ['site%s' % i for i in range(1, 11)]
    train_sessions = train_df[sites].fillna(0).astype('int').apply(lambda row: 
                                                     ' '.join([id2site[i] for i in row]), axis=1).tolist()
    test_sessions = test_df[sites].fillna(0).astype('int').apply(lambda row: 
                                                     ' '.join([id2site[i] for i in row]), axis=1).tolist()
    # we'll tell TfidfVectorizer that we'd like to split data by whitespaces only 
    # so that it doesn't split by dots (we wouldn't like to have 'mail.google.com' 
    # to be split into 'mail', 'google' and 'com')
    vectorizer = TfidfVectorizer(**vectorizer_params)
    X_train = vectorizer.fit_transform(train_sessions)
    X_test = vectorizer.transform(test_sessions)
    y_train = train_df['target'].astype('int').values
    
    # we'll need site visit times for further feature engineering
    train_times, test_times = train_df[times], test_df[times]
    
    # add scaled duration
    train_durations = (train_times.max(axis=1) - train_times.min(axis=1)).astype('timedelta64[ms]').astype('int')
    test_durations = (test_times.max(axis=1) - test_times.min(axis=1)).astype('timedelta64[ms]').astype('int')
    scaler = StandardScaler()
    train_dur_scaled = scaler.fit_transform(train_durations.values.reshape(-1, 1))
    test_dur_scaled = scaler.transform(test_durations.values.reshape(-1, 1))
    
    X_train = hstack([X_train, train_dur_scaled])
    X_test = hstack([X_test, test_dur_scaled])
    
    return X_train, X_test, y_train, vectorizer, train_times, test_times
    
    
def add_features(times, X_sparse):
    hour = times['time1'].apply(lambda ts: ts.hour)
    day_of_week = times['time1'].apply(lambda t: t.weekday()).values.reshape(-1, 1)
    day = ((hour >= 12) & (hour <= 18)).astype('int').values.reshape(-1, 1)
    alice_time = (times['time1'].apply(lambda ts: ts.hour in [9, 12, 13, 15, 16, 17, 18])).astype('int').values.reshape(-1,1)
    alice_dow = (times['time1'].apply(lambda t: t.weekday() not in [2, 5, 6])).astype('int').values.reshape(-1,1)
    month = times['time1'].apply(lambda t: t.month).values.reshape(-1, 1)
    year_month = times['time1'].apply(lambda t: 100 * t.year + t.month).values.reshape(-1, 1) / 1e5
    
    X = hstack([X_sparse, day_of_week, day, alice_time, alice_dow, year_month, month])
    return X
    
    
with timer('Building sparse site features'):
    X_train_sites, X_test_sites, y_train, vectorizer, train_times, test_times = \
        prepare_sparse_features(
            path_to_train=os.path.join(PATH_TO_DATA, 'train_sessions.csv'),
            path_to_test=os.path.join(PATH_TO_DATA, 'test_sessions.csv'),
            path_to_site_dict=os.path.join(PATH_TO_DATA, 'site_dic.pkl'),
            vectorizer_params={'ngram_range': SITE_NGRAMS,
                               'max_features': MAX_FEATURES,
                               'tokenizer': lambda s: s.split()})
                               
                               
with timer('Building additional features'):
    X_train_feat = add_features(train_times, X_train_sites)
    X_test_feat = add_features(test_times, X_test_sites)
    
    
with timer('Cross-validation'):
    time_split = TimeSeriesSplit(n_splits=NUM_TIME_SPLITS)
    logit = LogisticRegression(random_state=SEED, solver='liblinear')

    logit_grid_searcher = GridSearchCV(estimator=logit, param_grid={'C':[ LOGIT_C]},
                                  scoring='roc_auc', n_jobs=N_JOBS, cv=time_split, verbose=1)
    logit_grid_searcher.fit(X_train_feat, y_train)
    print('CV score', logit_grid_searcher.best_score_)
    
    
with timer('Test prediction and submission'):
    test_pred = logit_grid_searcher.predict_proba(X_test_feat)[:, 1]
    pred_df = pd.DataFrame(test_pred, index=np.arange(1, test_pred.shape[0] + 1),
                       columns=['target'])
    pred_df.to_csv(f'submission_alice_{AUTHOR}.csv', index_label='session_id')
   
