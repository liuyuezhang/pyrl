import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import re
import os


def get_plot_data(path, span=100):
    df = pd.DataFrame()

    with open(path + 'test.txt') as file:
        data = pd.read_csv(file, index_col=None)
        df = df.append(data, ignore_index=True)

    df['r'] = df['r'].ewm(span=span).mean()
    return df


i = 4

TIMESTEP = 1e6
NSAMPLE = 1e4
GAMES = ['Breakout', 'Seaquest', 'Pong', 'MontezumaRevenge', 'BitFlip']
YMAXS = [600, 2000, 5000, 1, 1, 6000, 17000, 1, 1]
METHODS = ['dqn', 'her-dqn']

res_dir = './res/'
files = os.listdir(res_dir)
sample_list = np.arange(0, TIMESTEP, TIMESTEP/NSAMPLE, dtype=np.int)

df = pd.DataFrame()
for file in os.listdir(res_dir):
    m = re.match('(.*)_(.*)_(.*)', file)
    env = m.group(1)
    method = m.group(2)
    seed = m.group(3)
    if (GAMES[i] in env) and (method in METHODS):
        path = res_dir + file + '/'
        data = get_plot_data(path)

        sample = pd.DataFrame()
        sample['t'] = sample_list
        sample['r'] = np.nan

        # interpolation
        res = pd.concat([sample, data], join='inner')
        res.sort_values('t', inplace=True)
        res.interpolate(method='linear', inplace=True)
        res = res[res['t'].isin(sample_list)]

        res['seed'] = int(seed)
        res[''] = method
        df = df.append(res, ignore_index=True)

print(df)

