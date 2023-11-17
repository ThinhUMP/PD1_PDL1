import pandas as pd
def target_bin(data, thresh, Y_name):
    t1 = data[Y_name] < thresh
    data.loc[t1, Y_name] = 0
    t2 = data[Y_name] >= thresh
    data.loc[t2, Y_name] = 1
    data[Y_name] = data[Y_name].astype('int64')
    return data