import pandas as pd
import math

df = pd.read_csv('data.csv')
df_train = df[0:math.floor(0.8*len(df))]
df_test = df[math.floor(0.8*len(df)):]
print("Training Data : ")
print(df_train.head(2))
print("Testing Data : ")
print(df_test.head(2))
df_train['text'].to_csv('train.csv', index=False)
df_test['text'].to_csv('test.csv', index=False)