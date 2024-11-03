import pandas as pd

# Load the dataset
data = pd.read_csv("Reddit_Data.csv")

# Inspect the first few rows of the data
print(data.head())
print(data.columns)
