import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load data
df = pd.read_csv("Titanic-Dataset.csv")

# Fill missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Convert text to numbers
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col].astype(str))

# Scale data
scaler = StandardScaler()
df = scaler.fit_transform(df)

print("Task Completed ✅")