import numpy as np, pandas as pd, os
from dateutil import parser
from utils    import *
import datetime

data_path = "../Data/AAPL_05222012_0930_1300_LOB_2.csv"

def ProcessColumnNames(df):
    """ Convert long column names to short column name for convenience
    E.g. TVITCH_41::AAPL.BID_PRICE1..TVITCH_41__AAPL_3 --> BID_PRICE1
    """
    new_col_names =  [i.split('.')[1] for i in df.columns[2:]]
    return list(df.columns[:2]) + new_col_names

def ProcessData(data_path = data_path):
    """ Read in the Level 2 Order Book data. Change the column names. And
    convert time string into numpy time object
    """
    processed_data_path = "../Data/df.pkl"
    if os.path.exists(processed_data_path):
        logger.info("Loading Processed Data")
        return pd.read_pickle(processed_data_path)
    else:
        logger.info("Loading Raw Data from path %s", data_path)
        df = pd.read_csv(data_path)
        logger.info("Change column names and time format")
        df.columns = ProcessColumnNames(df) 
        col = df.columns
        col = [i for i in col if 'TIME' not in i]
        df = df[col] # Remove time columns
        df['Time'] = [parser.parse(this_time) for this_time in df.Time]
        df.to_pickle(processed_data_path) 
        return df

def GetMid(df):
    return (df.ASK_PRICE1 + df.BID_PRICE1)/2

def GetMicro(df):
    return (df.ASK_PRICE1 * df.BID_SIZE1 + df.BID_PRICE1 * df.ASK_SIZE1) /\
           (df.ASK_SIZE1 + df.BID_SIZE1)

def GetMidChangeTime(df, time_horizon = 1):
    """ Get the change in mid price w.r.t. some time horizon (in second). 
    E.g. time_horizo of 1 seconds. 
    """
    logger.info("Getting Mid Change for Time %s seconds", time_horizon)
    save_path = "../Y/dmid_%gs.npy" %time_horizon
    index = df.Time.searchsorted(df.Time + datetime.timedelta(seconds =
        time_horizon), side = 'right') - 1
    mid = GetMid(df)
    dmid = np.array(mid[index]) - np.array(mid)
    if not os.path.exists(save_path):
        np.save(save_path, dmid)
    return dmid

def GetCrossingTime(df, time_horizon = 1):
    save_path = "../Y/Cross%gs.npy" %time_horizon
    index = df.Time.searchsorted(df.Time + datetime.timedelta(seconds = 
        time_horizon), side = 'right') - 1
    res = np.zeros(len(df), dtype = np.int)
    res[df.ASK_PRICE1[index].as_matrix() < df.BID_PRICE1.as_matrix()] = -1
    res[df.BID_PRICE1[index].as_matrix() > df.ASK_PRICE1.as_matrix()] = +1
    if not os.path.exists(save_path):
        np.save(save_path, res)
    return res

def GetMidChangeMessageClock(df, dg, mess_horizon):
    pass      

def GetBookPressure(df):
    logger.info("Getting Bid Ask Ratio")
    res = np.zeros((len(df), 2))
    res[:,0] = (df.BID_SIZE1 - df.ASK_SIZE1)/(df.BID_SIZE1 + df.ASK_SIZE1 +
            1)
    res[:,1] = np.log(df.BID_SIZE1 / df.ASK_SIZE1)
    np.save('../Features/BP.npy', res)
    logger.info(ListToString(Corr(GetMidChangeTime(df), res)))

half_lifes = [10, 20, 40, 80, 160, 320]
def GetEMAMid(df):
    logger.info("Getting EMA(Mid) - Mid features")
    res = np.zeros((len(df), len(half_lifes)))
    mid = GetMid(df) 
    for i in xrange(len(half_lifes)):
        res[:,i] = mid - mid.ewm(halflife = half_lifes[i]).mean() 
    np.save('../Features/EMAMid.npy', res)
    logger.info(ListToString(Corr(GetMidChangeTime(df), res)))

def GetBollinger(df):
    logger.info("Getting Bollinger Band Features") 
    res = np.zeros((len(df), len(half_lifes)*2))
    mid = GetMid(df)
    for i in xrange(len(half_lifes)):
        res[:, 2*i    ] = mid.ewm(halflife = half_lifes[i]).mean() +\
                            2 * mid.ewm(halflife = half_lifes[i]).std() - mid

        res[:, 2*i + 1] = mid.ewm(halflife = half_lifes[i]).mean() -\
                            2 * mid.ewm(halflife = half_lifes[i]).std() - mid
    logger.info(ListToString(Corr(GetMidChangeTime(df), res)))
    np.save('../Features/Bollinger.npy', res)

def GetMACD(df):
    logger.info("Getting MACD features")
    res = np.zeros((len(df), len(half_lifes) - 1))
    mid = GetMid(df)
    for i in xrange(len(half_lifes) - 1):
        macd = mid.ewm(halflife = half_lifes[i    ]).mean() -\
               mid.ewm(halflife = half_lifes[i + 1]).mean()
        res[:,i] = macd.ewm(halflife = .75*half_lifes[i]).mean() - macd
    logger.info(ListToString(Corr(GetMidChangeTime(df, 1), 
                        res)))
    np.save('../Features/MACD.npy', res)

def GetMessageData():
    dg = pd.read_csv('../Data/AAPL_05222012_0930_1300_message.csv')
    col = ['Time', 
            'TVITCH_41::AAPL.MESSAGE_TYPE', 'TVITCH_41::AAPL.PRICE',
            'TVITCH_41::AAPL.BUY_SELL_FLAG', 'TVITCH_41::AAPL.SIZE']
    dg = dg[col]
    col = [i.split('.')[-1] for i in col]
    col[col.index('MESSAGE_TYPE')] = 'MT'
    col[col.index('BUY_SELL_FLAG')] = 'BS'
    dg.columns = col
    dg['Time'] = [parser.parse(this_time) for this_time in dg.Time]
    return dg

if __name__ == '__main__':
    df = ProcessData()
    col = ['Time', 'BID_PRICE1', 'BID_SIZE1', 'ASK_PRICE1', 'ASK_SIZE1']
    dmid = GetMidChangeTime(df, 0.5)
    GetBookPressure(df)
    GetEMAMid(df)
    GetBollinger(df)
    GetMACD(df)
