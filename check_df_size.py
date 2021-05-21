import pandas as pd

df = pd.read_csv('data/dataset.csv', index_col=0)
df_group_by = df.groupby(['category']).count()

print(df_group_by)
