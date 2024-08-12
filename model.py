import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the dataset
data = pd.read_csv('Training.csv')

# Remove the unnecessary column
if 'Unnamed: 133' in data.columns:
    data = data.drop('Unnamed: 133', axis=1)

# Encode the target labels
encoder = LabelEncoder()
data['prognosis'] = encoder.fit_transform(data['prognosis'])

# Splitting the dataset into features and target variable
X = data.drop(columns=['prognosis'])
y = data['prognosis']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Train the Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Save the models and encoder
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('nb_model.pkl', 'wb') as f:
    pickle.dump(nb_model, f)

with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
