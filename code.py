import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load datasets
house_df = pd.read_csv("data/house_prices.csv")
medical_df = pd.read_csv("data/medical_appointments.csv")

# ---------------- CHECK MISSING VALUES ---------------- #

print("House Prices Missing Values:\n", house_df.isnull().sum())
print("\nMedical Dataset Missing Values:\n", medical_df.isnull().sum())

# ---------------- VISUALIZATION ---------------- #

house_df.isnull().sum().plot(kind='bar', figsize=(12,4))
plt.title("Missing Values in House Prices Dataset")
plt.show()

medical_df.isnull().sum().plot(kind='bar', figsize=(12,4))
plt.title("Missing Values in Medical Appointment Dataset")
plt.show()

# ---------------- HOUSE PRICE DATA CLEANING ---------------- #

num_cols = house_df.select_dtypes(include=np.number).columns
for col in num_cols:
    median_val = house_df[col].median()
    if not np.isnan(median_val):
        house_df[col] = house_df[col].fillna(median_val)

cat_cols = house_df.select_dtypes(include='object').columns
for col in cat_cols:
    if not house_df[col].mode().empty:
        house_df[col] = house_df[col].fillna(house_df[col].mode()[0])

# ---------------- MEDICAL DATA CLEANING ---------------- #

num_cols = medical_df.select_dtypes(include=np.number).columns
for col in num_cols:
    mean_val = medical_df[col].mean()
    if not np.isnan(mean_val):
        medical_df[col] = medical_df[col].fillna(mean_val)

cat_cols = medical_df.select_dtypes(include='object').columns
for col in cat_cols:
    if not medical_df[col].mode().empty:
        medical_df[col] = medical_df[col].fillna(medical_df[col].mode()[0])

# ---------------- REMOVE HIGH MISSING COLUMNS ---------------- #

threshold = 0.4
house_df = house_df.loc[:, house_df.isnull().mean() < threshold]
medical_df = medical_df.loc[:, medical_df.isnull().mean() < threshold]

# ---------------- FINAL VALIDATION ---------------- #

print("\nFinal House Prices Shape:", house_df.shape)
print("Final Medical Dataset Shape:", medical_df.shape)

print("\nFinal Missing Values (House):\n", house_df.isnull().sum())
print("\nFinal Missing Values (Medical):\n", medical_df.isnull().sum())

# ---------------- SAVE CLEANED DATA ---------------- #

os.makedirs("cleaned_data", exist_ok=True)

house_df.to_csv("cleaned_data/house_prices_cleaned.csv", index=False)
medical_df.to_csv("cleaned_data/medical_appointments_cleaned.csv", index=False)
