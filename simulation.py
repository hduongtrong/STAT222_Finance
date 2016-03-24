import numpy as np

def Sim(signal, thres, ask, bid):
    assert len(signal) == len(ask)
    assert len(signal) == len(bid)
    n = len(signal)
    pos = np.zeros(n, dtype = np.int)
    
    for i in xrange(n - 1):
        if signal[i] > +thres:
            pos[i] = 1
        elif signal[i] < -thres:
            pos[i] = -1
        else:
            pos[i] = pos[i - 1]

    trade = np.concatenate([[pos[0]], np.diff(pos)])

    pnl = np.zeros(n)
    pnl[0] = (bid[0] - ask[0]) * np.abs(pos[0])
    for i in xrange(1, n):
        if pos[i - 1] == 1:
            pnl[i] = bid[i] - bid[i - 1]
        elif pos[i - 1] == -1:
            pnl[i] = ask[i - 1] - ask[i]
        if (pos[i] != 0) and (pos[i] != pos[i - 1]):
            pnl[i] += bid[i] - ask[i]
    return pos, pnl, trade