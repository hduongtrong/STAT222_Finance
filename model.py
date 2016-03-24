import numpy as np, pandas as pd, os
import datetime
from sklearn.ensemble       import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble       import ExtraTreesRegressor,   ExtraTreesClassifier
from sklearn.linear_model   import LogisticRegressionCV, LassoCV
from sklearn.svm            import SVC, SVR
np.random.seed(1)

### 1. Setting Up
##### 1.1. Features
features = [
    '../Features/BP.npy',
    '../Features/EMAMid.npy',
    '../Features/MACD.npy',
    '../Features/Bollinger.npy',
    '../Features/Time_sensitive_v6_first_5.npy',
]

##### 1.2. What Output
y_file = '../Y/dmid_0.1s.npy'

##### 1.3. What Model
# model = ExtraTreesRegressor()
# model = LassoCV()
model = SVR()

### 2. Defining Functions and Parameters for Model
params = \
{
    'ExtraTreesRegressor': 
    {
        'n_estimators'          : 100           ,
        'max_depth'             : 10            ,
        'n_jobs'                : -1            ,
        'random_state'          : 1             ,
        'verbose'               : 2             ,
    },
    'ExtraTreesClassifier':
    {
        'n_estimators'          : 100           ,
        'max_depth'             : 10            ,
        'n_jobs'                : -1            ,
        'random_state'          : 1             ,
        'verbose'               : 2             ,
    },
    'LogisticRegressionCV':
    {
        'Cs'                    : 100           ,
        'cv'                    : 5             ,
        'multi_class'           : 'multinomial' ,
        'random_state'          : 1             ,
        'verbose'               : 1             ,
    },
    'LassoCV':
    {
        'cv'                    : 5             ,
        'random_state'          : 1             ,
        'verbose'               : True          ,
        'n_jobs'                : -1            ,
        'normalize'             : True          ,
    },
    'SVR':
    {
        'kernel'                : 'rbf'         ,
        'gamma'                 : 'auto'        ,
        'C'                     : 1.0           ,
        'verbose'               : 2             ,
    },
}



def GetAllFeatures(features = features):
    X = []
    for feature in features:
        print("Loading %s feature" % feature.split('/')[-1])
        if feature.endswith('npy'):
            X.append(np.load(feature))
        elif feature.endswith('csv'):
            X.append(pd.read_csv(feature).as_matrix())
        else:
            print("Feature set need to be either csv or npy file")
    return np.hstack(X)

def GetData():
    y = np.load('../Y/dmid_0.1s.npy')
    X = GetAllFeatures(features)
    time = pd.read_pickle('../Data/df.pkl')['Time']
    # Index of the time at 11AM. We only use data from 930 to 1100.
    T11am = time.searchsorted(time[0] + 
            datetime.timedelta(seconds = 90*60))[0]
    T12pm = time.searchsorted(time[0] + 
            datetime.timedelta(seconds = 150*60))[0]
    # Use 50% for Train, 25% for Valid, and 25% for test
    train = np.arange(100, T11am / 2, dtype = np.int)
    valid = np.arange(T11am / 2, T11am * 3 / 4, dtype = np.int)
    test  = np.arange(T11am * 3 / 4, T11am, dtype = np.int)
    return X, y, train, valid, test

if __name__ == '__main__':
    try:
        X
    except NameError:
        X, y, train, valid, test = GetData()
    model.set_params(**params[model.__class__.__name__])
    model.fit(X[train[::2]], y[train[::2]])
    print model.score(X[valid], y[valid])
