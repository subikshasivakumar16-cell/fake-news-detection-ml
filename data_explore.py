import pandas as pd

# Load fake and real news datasets
fake = pd.read_csv("dataset/Fake.csv")
real = pd.read_csv("dataset/True.csv")

# Add labels: 0 = Fake, 1 = Real
fake['label'] = 0
real['label'] = 1

# Combine datasets
data = pd.concat([fake, real], ignore_index=True)

# Shuffle data
data = data.sample(frac=1, random_state=42)

# Quick look
print("First 5 rows:\n", data.head())
print("\nDataset shape:", data.shape)
print("\nClass distribution:\n", data['label'].value_counts())
