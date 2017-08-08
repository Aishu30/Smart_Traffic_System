import matplotlib.pyplot as plt
import pandas as pd
from sklearn import KMeans

data = pd.read_csv("traffic.csv", header = None)
print(data.head())

