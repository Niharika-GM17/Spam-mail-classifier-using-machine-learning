import pandas as pd

# Load CSV
df = pd.read_csv('spam.csv', encoding='latin-1')

# Keep only relevant columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Quick look
print(df.head())
print(df['label'].value_counts())
