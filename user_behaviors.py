# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from scipy import stats

# Step 2: Load and Clean Dataset
df = pd.read_csv("user_behavior_dataset.csv")  # Ensure this file is in your working directory
df.dropna(inplace=True)  # Remove rows with missing values
df = df[df['Screen_On_Time'] > 0]  # Filter out invalid screen time entries

# Step 3: Descriptive Statistics
mean_screen = df['Screen_On_Time'].mean()
median_screen = df['Screen_On_Time'].median()
mode_screen = df['Screen_On_Time'].mode()[0]
std_screen = df['Screen_On_Time'].std()
range_screen = df['Screen_On_Time'].max() - df['Screen_On_Time'].min()

print("📊 Descriptive Statistics for Screen-On Time")
print(f"Mean: {mean_screen:.2f} hours/day")
print(f"Median: {median_screen:.2f}")
print(f"Mode: {mode_screen:.2f}")
print(f"Standard Deviation: {std_screen:.2f}")
print(f"Range: {range_screen:.2f}\n")

# Step 4: Visualisations
plt.figure(figsize=(8,5))
sns.histplot(df['Screen_On_Time'], bins=20, kde=True, color='teal')
plt.title("Distribution of Screen-On Time")
plt.xlabel("Hours per Day")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x=df['App_Usage_Time'], color='orange')
plt.title("Boxplot of App Usage Time")
plt.xlabel("Minutes per Day")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Variables")
plt.tight_layout()
plt.show()

# Step 5: Grouped Analysis
os_group = df.groupby('Operating_System')['Screen_On_Time'].mean()
os_group.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title("Average Screen-On Time by Operating System")
plt.ylabel("Hours per Day")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Step 6: Summary Table
print("📋 Summary Table of Dataset")
print(df.describe())
