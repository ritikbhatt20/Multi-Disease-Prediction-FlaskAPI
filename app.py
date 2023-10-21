import pickle

import numpy as np
from flask import Flask, request, jsonify

modelDiabetes = pickle.load(open('diabetes_model.pkl', 'rb'))
modelHeart = pickle.load(open('heart_disease_model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return "Hello World here"


@app.route('/predictD', methods=['POST'])
def predict():

    # Inputs for Diabetes model

    Pregnancies = request.form.get('Pregnancies')
    Glucose = request.form.get('Glucose')
    BloodPressure = request.form.get('BloodPressure')
    SkinThickness = request.form.get('SkinThickness')
    Insulin = request.form.get('Insulin')
    BMI = request.form.get('BMI')
    DiabetesPedigreeFunction = request.form.get('DiabetesPedigreeFunction')
    Age = request.form.get('Age')

    age = request.form.get('age')
    sex = request.form.get('sex')
    cp = request.form.get('cp')
    trestbps = request.form.get('trestbps')
    chol = request.form.get('chol')
    fbs = request.form.get('fbs')
    restecg = request.form.get('restecg')
    thalach = request.form.get('thalach')
    exang = request.form.get('exang')
    oldpeak = request.form.get('oldpeak')
    slope = request.form.get('slope')
    ca = request.form.get('ca')
    thal = request.form.get('thal')

    input_query_diabetes = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]], dtype=np.float64)
    input_query_heart = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], dtype=np.float64)

    result_diabetes = modelDiabetes.predict(input_query_diabetes)[0]
    result_heart = modelHeart.predict(input_query_heart)[0]

    return jsonify({'Outcome of Diabetes': str(result_diabetes), 'Outcome of Heart': str(result_heart)})


if __name__ == '__main__':
    app.run(debug=True)
