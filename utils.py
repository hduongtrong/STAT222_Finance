import logging, pandas as pd, numpy as np

try:
    logger
except NameError:
    logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
            filename="/tmp/history.log",
            filemode='a', level=logging.DEBUG,
            datefmt='%m/%d/%y %H:%M:%S')
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
            datefmt='%m/%d/%y %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    logger = logging.getLogger(__name__)

pd.set_option('display.height', 200)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 200)


def GetNonNAIndex(x, y):
    return np.array(np.logical_and(np.logical_not(np.isnan(x)), 
                          np.logical_not(np.isnan(y))))
def corr(x, y): return np.corrcoef(x[GetNonNAIndex(x,y)], 
                                   y[GetNonNAIndex(x,y)])[0][1]
cor = corr
def head(df): return df.head()
def tail(df): return df.tail()

## For compliance with R
dim = np.shape
length = len

def Corr(y, yhats):
    return [corr(y, yhats[:,i]) for i in xrange(yhats.shape[1])]

def ListToString(l):
    return '|'.join(["%.2f" %i for i in l])


def table(x):
    assert np.issubdtype(x.dtype, np.integer)
    unique, counts = np.unique(x, return_counts = True)
    return np.asarray((unique, counts)).T

def table2(x):
    elements = np.unique(x)
    counts = {}
    for element in elements:
        counts[element] = np.sum(x == element)
    return counts
