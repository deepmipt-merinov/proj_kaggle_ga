import datetime
import json
import numpy as np
import pandas as pd


def load(file, usecols=None, nrows=None, unpack_json=True):

    JSON_COLUMNS = ['totals']

    if unpack_json:
        converters = {'converters': {column: json.loads for column in JSON_COLUMNS}}
    else:
        converters = {}

    # Read table, preserve accuracy
    tbl = pd.read_csv(file, usecols=usecols, **converters, dtype={'date': 'str', 'fullVisitorId': 'str'}, parse_dates=['date'], nrows=nrows)

    if unpack_json:
        for column in JSON_COLUMNS:
            # Read json column to table
            sub = pd.io.json.json_normalize(tbl[column])
            # Change column name
            sub.columns = ['{}.{}'.format(column, subcolumn) for subcolumn in sub.columns]
            # Drop old column
            tbl = tbl.drop(column, axis=1)
            # Join new table
            tbl = tbl.join(sub, how='left')

    return tbl


def rolling(d, k, s, h, A, B):
    """
    d : tr size
    k : te size
    s : stride size
    h : step size
    A :
    B :

    |--h--|-----d-----|--s--|---k---|
          |--h--|-----d-----|--s--|---k---|
    """
    results = []

    p = A
    while p < B-k-s:

        x1 = p
        x2 = p+d
        x3 = p+d+s
        x4 = p+d+s+k

        p = p+h
        r = (x1, x2, x3, x4)

        results.append(r)

    return results


def inverseRolling(d, k, s, h, A, B):
    """
    d : tr size
    k : te size
    s : stride size
    h : step size
    A :
    B :

    |--h--|-----d-----|--s--|---k---|
          |--h--|-----d-----|--s--|---k---|
    """
    results = []

    p = B
    while p > A+d+k+s:

        x4 = B
        x3 = B-k
        x2 = p-k-s
        x1 = p-k-s-d

        p = p-h
        r = (x1, x2, x3, x4)

        results.append(r)

    return results


def calculateMarkovTransitionMatrix(tbl, params, A, B):

    THRESHOLD = 1

    d = datetime.timedelta(days=params['d'])
    k = datetime.timedelta(days=params['k'])
    s = datetime.timedelta(days=params['s'])
    h = datetime.timedelta(days=params['h'])

    folds = inverseRolling(d, k, s, h, A, B)

    PMs = []
    CMs = []
    for fold in folds:

        print([x.strftime('%Y-%m-%d') for x in fold], flush=True)

        x0 = fold[0]
        x1 = fold[1]
        x2 = fold[2]
        x3 = fold[3]

        maskO = (tbl['date'] <= x0)
        maskA = (tbl['date'] >= x0) & (tbl['date'] <= x1)
        maskB = (tbl['date'] >= x2) & (tbl['date'] <= x3)

        tblO = tbl.loc[maskO]
        tblA = tbl.loc[maskA]
        tblB = tbl.loc[maskB]

        rO = tblO['fullVisitorId'].unique()
        rA = tblA.groupby('fullVisitorId')['totals.totalTransactionRevenue'].sum()
        rB = tblB.groupby('fullVisitorId')['totals.totalTransactionRevenue'].sum()

        am = np.array(
            [[0,0], [0,0], [0,0]]
        )
        cm = np.array(
            [[0,0], [0,0], [0,0]]
        )
        base = pd.concat([rA, rB], axis=1, join='outer')
        base.columns = ['A','B']
        base['old'] = base.index.isin(rO).astype(int)
        xs = np.column_stack(
            [base['A'].values, base['B'].values, base['old'].values]
        )
        for x in xs:

            if np.isnan(x[1]):  # already existed ID
                x[1] = 0.0

            if np.isnan(x[0]) and (x[2] == 0) and (x[1] < THRESHOLD):
                am[0][0] += 1
                cm[0][0] += 0
            if np.isnan(x[0]) and (x[2] == 0) and (x[1] > THRESHOLD):
                am[0][1] += 1
                cm[0][1] += x[1]
            if np.isnan(x[0]) and (x[2] == 1) and (x[1] < THRESHOLD):
                am[1][0] += 1
                cm[1][0] += 0
            if np.isnan(x[0]) and (x[2] == 1) and (x[1] > THRESHOLD):
                am[1][1] += 1
                cm[1][1] += x[1]
            if x[0] < THRESHOLD and x[1] < THRESHOLD:
                am[1][0] += 1
                cm[1][0] += 0
            if x[0] < THRESHOLD and x[1] > THRESHOLD:
                am[1][1] += 1
                cm[1][1] += x[1]
            if x[0] > THRESHOLD and x[1] < THRESHOLD:
                am[2][0] += 1
                cm[2][0] += 0
            if x[0] > THRESHOLD and x[1] > THRESHOLD:
                am[2][1] += 1
                cm[2][1] += x[1]

        # Estimate probability matrix
        pm = am / am.sum()
        # Estimate cost matrix (overestimate with Jensen's inequality) log(1/n sum x_i) >= sum 1/n log(x_i)
        cm = np.log1p(cm / (1+am))

        PMs.append(pm)
        CMs.append(cm)

    return PMs, CMs


def forecastTransactionRevenue(tbl, params, A, B, pm, cm):

    THRESHOLD = 1

    d = datetime.timedelta(days=params['d'])
    k = datetime.timedelta(days=params['k'])
    s = datetime.timedelta(days=params['s'])
    h = datetime.timedelta(days=params['h'])

    posteriorm = pm / pm.sum(axis=1, keepdims=True)

    print(posteriorm, flush=True)
    print('Expected log-revenue per user: {: .6f}'.format(posteriorm[0][1] * cm[0][1]))
    print('Expected log-revenue per user: {: .6f}'.format(posteriorm[1][1] * cm[1][1]))
    print('Expected log-revenue per user: {: .6f}'.format(posteriorm[2][1] * cm[2][1]))

    UNIQUE = tbl['fullVisitorId'].unique()  # global unique users
    forecast = {
        k: 0.0 for k in UNIQUE
    }
    folds = inverseRolling(d, k, s, h, A, B)
    for fold in folds:

        print([x.strftime('%Y-%m-%d') for x in fold], flush=True)

        x0 = fold[0]
        x1 = fold[1]
        x2 = fold[2]
        x3 = fold[3]

        maskO = (tbl['date'] <= x0)
        maskA = (tbl['date'] >= x0) & (tbl['date'] <= x1)

        tblO = tbl.loc[maskO]
        tblA = tbl.loc[maskA]
        tblT = tbl.loc[maskO | maskA]

        transactionsA = tblA.groupby('fullVisitorId')['totals.totalTransactionRevenue'].sum().to_dict()

        # Expectation on old users
        for k, c in transactionsA.items():
            if c < THRESHOLD:
                forecast[k] += posteriorm[1][1] * cm[1][1]
            else:
                forecast[k] += posteriorm[2][1] * cm[2][1]

        # Expectation on new users
        diff_ks = set(UNIQUE) - set(tblT['fullVisitorId'].unique())
        for k in diff_ks:
            forecast[k] += posteriorm[0][1] * cm[0][1]

    N = len(folds)
    for k in forecast.keys():
        forecast[k] /= N

    return forecast
