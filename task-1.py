import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load data
df = pd.read_csv("Titanic-Dataset.csv")

# -------------------------------
# 1. Original Data
# -------------------------------
print("🔹 Original Data (First 5 rows):")
print(df.head())

print("\n🔹 Dataset Shape:", df.shape)

# -------------------------------
# 2. Missing Values
# -------------------------------
print("\n🔹 Missing Values:")
print(df.isnull().sum())

# Fill missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# -------------------------------
# 3. Encoding
# -------------------------------
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col].astype(str))

print("\n🔹 After Encoding:")
print(df.head())

# -------------------------------
# 4. Feature Scaling
# -------------------------------
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

df = pd.DataFrame(df_scaled, columns=df.columns)

# -------------------------------
# 5. Final Preprocessed Data
# -------------------------------
print("\n🔹 Preprocessed Data (First 5 rows):")
print(df.head())

print("\n🔹 Final Shape:", df.shape)

# -------------------------------
print("\n✅ Task Completed Successfully!")