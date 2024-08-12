from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# Load models and encoder
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('nb_model.pkl', 'rb') as f:
    nb_model = pickle.load(f)

with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# Load the dataset to get symptom names
data = pd.read_csv('Training.csv')
if 'Unnamed: 133' in data.columns:
    data = data.drop('Unnamed: 133', axis=1)

symptoms = data.columns[:-1]

# Create a symptom index dictionary
symptom_index = {symptom: i for i, symptom in enumerate(symptoms)}

app = Flask(__name__)

def predict_disease(input_symptoms, model_choice):
    input_data = [0] * len(symptoms)
    for symptom in input_symptoms:
        if symptom in symptom_index:
            input_data[symptom_index[symptom]] = 1

    if model_choice == 'RandomForest':
        model = rf_model
    elif model_choice == 'NaiveBayes':
        model = nb_model

    prediction = model.predict([input_data])[0]
    disease = encoder.inverse_transform([prediction])[0]
    return disease

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_symptoms = [request.form['symptom1'], request.form['symptom2'], request.form['symptom3']]
        model_choice = request.form['model_choice']
        disease = predict_disease(selected_symptoms, model_choice)
        return render_template('index.html', symptoms=symptoms, disease=disease, selected_symptoms=selected_symptoms, model_choice=model_choice)
    return render_template('index.html', symptoms=symptoms)

if __name__ == '__main__':
    app.run(debug=True)
