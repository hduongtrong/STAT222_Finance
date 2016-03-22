import matplotlib.pyplot as plt, pandas as pd
from data import *
df = ProcessData()
dg = GetMessageData()

mic = GetMicro(df)
plt.plot_date(df.Time, df.ASK_PRICE1, ls = 'solid', color = 'red', 
        drawstyle = 'steps-post', fmt = 'bo', marker = '.')
plt.plot_date(df.Time, df.BID_PRICE1, ls = 'solid', color = 'green',
        drawstyle = 'steps-post', marker = '.')
plt.plot_date(df.Time, mic,           ls = 'solid', color = 'blue',
        drawstyle = 'steps-post', marker = '.')
