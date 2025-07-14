import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load your dataset (adjust path if needed)
df = pd.read_csv("cityData.csv")


# Clean and rename relevant columns
df.columns = df.columns.str.strip()
df = df.rename(columns={"population": "Population", "State": "State"})

# Fill in missing features with dummy/default values
df["RoadQuality"] = 50000   # default/fake value
df["Tier"] = 2              # 1=Top, 2=Intermediate, 3=Low
df["EconomyIndex"] = 3000   # estimated economic value
df["LiteracyRate"] = 7      # out of 10, just an estimate

# Drop any row missing key fields
df = df.dropna(subset=["Population", "State"])

# Define input features and label
X = df[["Population", "RoadQuality", "Tier", "EconomyIndex", "LiteracyRate"]]
y = df["State"]

# Train a basic classifier
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save the model to disk
with open("state_prediction.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as state_prediction.pkl")

