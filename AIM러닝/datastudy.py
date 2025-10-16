import pandas as pd
import numpy as np
from sklearn.datasets import load_wine

df, y = load_wine(as_frame=True, return_X_y=True)
print(df.head())
df["quality"] = y
# print(pd.Series(df["quality"]).nunique())
# malic_mean = df["malic_acid"].mean()
# print(malic_mean)
# high_color_ratio = sum(1 for ci in df["color_intensity"] if ci >= 10.0) / len(y)
# print(high_color_ratio)
class_data = pd.DataFrame({"y": y, "ash": df["ash"]}).groupby("y")["ash"].min().idxmin()
print(class_data)
