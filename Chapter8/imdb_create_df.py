import os
import pyprind
import pandas as pd
import numpy as np


basepath = '../data/aclImdb'
labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000)
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path, file),
                      'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]],
                           ignore_index=True)
            pbar.update()
df.columns = ['review', 'sentiment']

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('../data/movie_data.csv', index=False, encoding='utf-8')

df = pd.read_csv('../data/movie_data.csv', encoding='utf-8')
print(df.head(3))
print(df.shape)
