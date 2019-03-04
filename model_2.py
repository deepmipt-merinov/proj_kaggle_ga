import gc
import os
import sys
import json
import math
import time
import datetime
import itertools
import collections
import functools
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

import feather
import matplotlib
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd

from utils import load
from utils import inverseRolling
from utils import calculateMarkovTransitionMatrix
from utils import forecastTransactionRevenue


SHOULD_SAVE = False
DATA_DIR = '/Users/pashapashaa/shared/gstore/data/'

TR_FILE = os.path.join(DATA_DIR, 'train_v2.csv')
TE_FILE = os.path.join(DATA_DIR, 'test_v2.csv')
SM_FILE = os.path.join(DATA_DIR, 'sample_submission_v2.csv')

print(TR_FILE)
print(TE_FILE)
print(SM_FILE)

PARAMS_GRID = [
    {'d': 30, 'k': 60+1, 's': 45, 'h': 1},  # tr_size, te_size, gap_size, shift_size in days
    {'d': 35, 'k': 60+1, 's': 45, 'h': 1},
    {'d': 40, 'k': 60+1, 's': 45, 'h': 1},
    {'d': 45, 'k': 60+1, 's': 45, 'h': 1},
    {'d': 50, 'k': 60+1, 's': 45, 'h': 1},
    {'d': 55, 'k': 60+1, 's': 45, 'h': 1},
    {'d': 60, 'k': 60+1, 's': 45, 'h': 1}
]


if __name__ == '__main__':


    # SAVE OR LOAD DATASET

    if SHOULD_SAVE:

        # Load raw
        tr_tbl = load(TR_FILE, usecols=['fullVisitorId', 'date', 'totals'], nrows=None, unpack_json=True)
        te_tbl = load(TE_FILE, usecols=['fullVisitorId', 'date', 'totals'], nrows=None, unpack_json=True)

        # Subset columns
        selected = [
            'fullVisitorId',
            'date',
            'totals.totalTransactionRevenue'
        ]
        tr_tbl = tr_tbl[selected]
        te_tbl = te_tbl[selected]

        # Transform target
        tr_tbl.loc[:, 'totals.totalTransactionRevenue'] = tr_tbl['totals.totalTransactionRevenue'].astype(float)
        tr_tbl.loc[:, 'totals.totalTransactionRevenue'] = tr_tbl['totals.totalTransactionRevenue'].fillna(0.0)

        te_tbl.loc[:, 'totals.totalTransactionRevenue'] = te_tbl['totals.totalTransactionRevenue'].astype(float)
        te_tbl.loc[:, 'totals.totalTransactionRevenue'] = te_tbl['totals.totalTransactionRevenue'].fillna(0.0)

        # Convert types
        types = [
            ('fullVisitorId', 'str'), ('date', 'str'), ('totals.totalTransactionRevenue', 'float')
        ]
        for k, v in types:
            tr_tbl.loc[:,k] = tr_tbl[k].astype(v)
            te_tbl.loc[:,k] = te_tbl[k].astype(v)

        # Save to feather
        tr_tbl.to_feather(os.path.join(DATA_DIR, 'tr.feather'))
        te_tbl.to_feather(os.path.join(DATA_DIR, 'te.feather'))

        assert tr_tbl.equals(feather.read_dataframe(os.path.join(DATA_DIR, 'tr.feather')))
        assert te_tbl.equals(feather.read_dataframe(os.path.join(DATA_DIR, 'te.feather')))
        print('Everything Ok!')

    else:
        tr_tbl = feather.read_dataframe(os.path.join(DATA_DIR, 'tr.feather'))
        te_tbl = feather.read_dataframe(os.path.join(DATA_DIR, 'te.feather'))


    tr_tbl.loc[:, 'date'] = pd.to_datetime(tr_tbl['date'], format='%Y-%m-%d')
    te_tbl.loc[:, 'date'] = pd.to_datetime(te_tbl['date'], format='%Y-%m-%d')


    # LEARN TRANSITION AND COST MATRICIES

    PM_grid2016 = []  # prob matricies
    CM_grid2016 = []  # cost matricies
    PM_grid2017 = []  # prob matricies
    CM_grid2017 = []  # cost matricies

    for params in PARAMS_GRID:

        print('\n\nLearning params: {}...\n'.format(params), flush=True)

        # Process 2016 year
        A = datetime.datetime(2016, 7, 31)
        B = datetime.datetime(2017, 1, 31)
        PMs, CMs = calculateMarkovTransitionMatrix(tr_tbl, params, A, B)

        PM = np.median(np.stack(PMs, axis=0), axis=0)
        PM = PM / PM.sum()
        CM = np.median(np.stack(CMs, axis=0), axis=0)

        PM_grid2016.append(PM)
        CM_grid2016.append(CM)

        print(PM / PM.sum(axis=1, keepdims=True))
        print(CM)
        print(flush=True)

        # Process 2017 year
        A = datetime.datetime(2017, 7, 31)
        B = datetime.datetime(2018, 1, 31)
        PMs, CMs = calculateMarkovTransitionMatrix(tr_tbl, params, A, B)

        PM = np.median(np.stack(PMs, axis=0), axis=0)
        PM = PM / PM.sum()
        CM = np.median(np.stack(CMs, axis=0), axis=0)

        PM_grid2017.append(PM)
        CM_grid2017.append(CM)

        print(PM / PM.sum(axis=1, keepdims=True))
        print(CM)
        print(flush=True)


    # FORECAST TRANSACTIONS REVENUE

    forecasts_grid2016 = []
    forecasts_grid2017 = []

    for params, PM, CM in zip(PARAMS_GRID, PM_grid2016, CM_grid2016):
        print('\n\nForecasting with params: {}...\n'.format(params), flush=True)
        A = datetime.datetime(2018, 7, 31)
        B = datetime.datetime(2019, 1, 31)
        forecasts = forecastTransactionRevenue(te_tbl, params, A, B, PM, CM)
        forecasts_grid2016.append(forecasts)

    for params, PM, CM in zip(PARAMS_GRID, PM_grid2017, CM_grid2017):
        print('\n\nForecasting with params: {}...\n'.format(params), flush=True)
        A = datetime.datetime(2018, 7, 31)
        B = datetime.datetime(2019, 1, 31)
        forecasts = forecastTransactionRevenue(te_tbl, params, A, B, PM, CM)
        forecasts_grid2017.append(forecasts)

    # Extract unique users
    users = te_tbl['fullVisitorId'].unique()
    # Revenue for 2016 year
    R2016 = {k: 0.0 for k in users}
    # Revenue for 2017 year
    R2017 = {k: 0.0 for k in users}

    N2016_grid = len(forecasts_grid2016)
    N2017_grid = len(forecasts_grid2017)
    for user in users:
        for forecasts2016, forecasts2017 in zip(forecasts_grid2016, forecasts_grid2017):
            R2016[user] += forecasts2016[user] / N2016_grid  # average with fixed weights
            R2017[user] += forecasts2017[user] / N2017_grid  # average with fixed weights

    # Aggregate 2016 and 2017 years
    R = {}
    for user in users:
        R[user] = 0.3*R2016[user] + 0.7*R2017[user]

    pl.figure()
    pl.hist([v for v in R2016.values()], bins=50, log=True, color='r', alpha=0.8)
    pl.show()

    pl.figure()
    pl.hist([v for v in R2017.values()], bins=50, log=True, color='b', alpha=0.8)
    pl.show()

    pl.figure()
    pl.hist([v for v in R.values()], bins=50, log=True, color='m', alpha=0.8)
    pl.show()


    # MAKE FINAL SUBMISSION

    submission = pd.read_csv(SM_FILE, dtype={'fullVisitorId': 'str'}, index_col='fullVisitorId', squeeze=True)

    if set(submission.index) - set([k for k in R]):
        print('Failed to extract users from dataset [submission users - R users]', flush=True)
    if set([k for k in R]) - set(submission.index):
        print('Failed to extract users from dataset [R users - submission users]', flush=True)

    submission.update(pd.Series(R))
    submission.to_csv('../submit-29-11-2018-new.csv', index=True, header=True, float_format='%.6f')

    print('WE ARE FUCKING DONE!', flush=True)
