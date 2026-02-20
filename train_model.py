import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import pickle

# Load dataset
df = pd.read_csv('phishing.csv')

# Drop Index column, split features and label
X = df.drop(['Index', 'class'], axis=1)
y = df['class']

print("Number of features:", X.shape[1])  # Should print 30

# Train model
gbc = GradientBoostingClassifier(max_depth=4, learning_rate=0.7, max_features='sqrt',
                                  n_estimators=200, subsample=0.8, random_state=42)
gbc.fit(X, y)

# Save new model
with open('pickle/model.pkl', 'wb') as file:
    pickle.dump(gbc, file)

print("Model trained and saved successfully!")
