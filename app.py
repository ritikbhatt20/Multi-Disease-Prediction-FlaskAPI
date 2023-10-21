import pickle

import numpy as np
from flask import Flask, request, jsonify

model = pickle.load(open('diabetes_model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return "Hello World here"


@app.route('/predictD', methods=['POST'])
def predict():
    Pregnancies = request.form.get('Pregnancies')
    Glucose = request.form.get('Glucose')
    BloodPressure = request.form.get('BloodPressure')
    SkinThickness = request.form.get('SkinThickness')
    Insulin = request.form.get('Insulin')
    BMI = request.form.get('BMI')
    DiabetesPedigreeFunction = request.form.get('DiabetesPedigreeFunction')
    Age = request.form.get('Age')

    input_query = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]], dtype=np.float64)

    result = model.predict(input_query)[0]

    return jsonify({'Outcome of Diabetes': str(result)})


if __name__ == '__main__':
    app.run(debug=True)
