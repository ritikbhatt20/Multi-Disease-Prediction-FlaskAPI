import pickle

import numpy as np
from flask import Flask, request, jsonify

model = pickle.load(open('diabetes_model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return "Hello World here"


@app.route('/predict', methods=['POST'])
def predict():
    Pregnancies = request.form.get('Pregnancies')
    cgpa = request.form.get('cgpa')
    cgpa = request.form.get('cgpa')
    cgpa = request.form.get('cgpa')
    cgpa = request.form.get('cgpa')
    cgpa = request.form.get('cgpa')
    cgpa = request.form.get('cgpa')
    cgpa = request.form.get('cgpa')


if __name__ == '__main__':
    app.run(debug=True)
