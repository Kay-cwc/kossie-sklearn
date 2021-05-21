import pandas as pd
import numpy as np


alias = {
    'relationship': ['relationship_advice'],
    'sex': ['sex'],
    'mentalHealth': ['mentalHealth'],
    'career': ['findapath', 'careerguidance'],
}


def open_csv(alias):
    base_dir = '../webcrawl/data'
    df = pd.read_csv('{}/{}.csv'.format(base_dir, alias), index_col=0)
    return df


index = 0
for idx, (key, val) in enumerate(alias.items()):
    for d in val:
        if index == 0:
            df = open_csv(d)
            df['category'] = key
            print(df.shape)
        else:
            df_new = open_csv(d)
            df_new['category'] = key
            print(df_new.shape)
            df = pd.concat([df, df_new], ignore_index=True)
        index += 1

print(df)
df.to_csv('data/dataset.csv')
