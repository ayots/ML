# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mode

# Create a messy dataset
data = {
    'User_ID': [101, 102, 103, 104, 105, 105],
    'Screen_On_Time': [4.5, np.nan, 6.0, 3.5, 4.5, 4.5],
    'App_Usage_Time': [120, 150, np.nan, 90, 120, 120],
    'Battery_Drain': [3000, 3200, 3100, np.nan, 3000, 3000],
    'Operating_System': ['Android', 'iOS', 'Android', 'iOS', 'Android', 'Android']
}
df_messy = pd.DataFrame(data)

# 🔍 Visualize Messy Dataset
plt.figure(figsize=(8,5))
plt.title("Messy Data: Screen-On Time")
plt.xlabel("Index")
plt.ylabel("Hours per Day")
plt.plot(df_messy['Screen_On_Time'], marker='o', linestyle='--', color='red')
plt.grid(True)
plt.tight_layout()
plt.show()

# 🧼 Clean the dataset
df_clean = df_messy.drop_duplicates()
df_clean['Screen_On_Time'].fillna(df_clean['Screen_On_Time'].mean(), inplace=True)
df_clean['App_Usage_Time'].fillna(df_clean['App_Usage_Time'].median(), inplace=True)
df_clean['Battery_Drain'].fillna(mode(df_clean['Battery_Drain'].dropna()), inplace=True)

# ✅ Visualize Cleaned Dataset
plt.figure(figsize=(8,5))
plt.title("Cleaned Data: Screen-On Time")
plt.xlabel("Index")
plt.ylabel("Hours per Day")
plt.plot(df_clean['Screen_On_Time'], marker='o', linestyle='-', color='green')
plt.grid(True)
plt.tight_layout()
plt.show()
