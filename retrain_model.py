import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump


# Load your cleaned dataset
df = pd.read_csv('data_cleaned.csv')  # Make sure this file is in the same directory

# Preprocess the state column
df['state'] = df['state'].str.replace(' ', '')  # Ensure consistency with your app

# Define the correct feature columns based on your data
features = ['population', 'dist_road_qual', 'tier_value', 'edi', 'literacy_rate']
target = 'state'

# Prepare data
X = df[features]
y = df[target]

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save the trained model
dump(model, 'state_prediction.joblib')

print("Model trained and saved as 'state_prediction.joblib'")
