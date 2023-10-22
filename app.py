import pickle

import numpy as np
from flask import Flask, request, jsonify

modelDiabetes = pickle.load(open('diabetes_model.pkl', 'rb'))
modelHeart = pickle.load(open('heart_disease_model.pkl', 'rb'))
modelParkinson = pickle.load(open('parkinsons_model.pkl', 'rb'))

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

    # Inputs for Heart model

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

    # Inputs for Parkinson's model

    MDVP_Fo_Hz = request.form.get('MDVP:Fo(Hz)')
    MDVP_Fhi_Hz = request.form.get('MDVP:Fhi(Hz)')
    MDVP_Flo_Hz = request.form.get('MDVP:Flo(Hz)')
    MDVP_Jitter_per = request.form.get('MDVP:Jitter(%)')
    MDVP_Jitter_Abs = request.form.get('MDVP:Jitter(Abs)')
    MDVP_RAP = request.form.get('MDVP:RAP')
    MDVP_PPQ = request.form.get('MDVP:PPQ')
    Jitter_DDP = request.form.get('Jitter:DDP')
    MDVP_Shimmer = request.form.get('MDVP:Shimmer')
    MDVP_Shimmer_dB = request.form.get('MDVP:Shimmer(dB)')
    Shimmer_APQ3 = request.form.get('Shimmer:APQ3')
    Shimmer_APQ5 = request.form.get('Shimmer:APQ5')
    MDVP_APQ = request.form.get('MDVP:APQ')
    Shimmer_DDA = request.form.get('Shimmer:DDA')
    NHR = request.form.get('NHR')
    HNR = request.form.get('HNR')
    RPDE = request.form.get('RPDE')
    DFA = request.form.get('DFA')
    spread1 = request.form.get('spread1')
    spread2 = request.form.get('spread2')
    D2 = request.form.get('D2')
    PPE = request.form.get('PPE')

    input_query_diabetes = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]], dtype=np.float64)
    input_query_heart = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], dtype=np.float64)
    input_query_parkinson = np.array([[MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter_per, MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]], dtype=np.float64)

    result_diabetes = modelDiabetes.predict(input_query_diabetes)[0]
    result_heart = modelHeart.predict(input_query_heart)[0]
    result_parkinson = modelParkinson.predict(input_query_parkinson)[0]

    return jsonify({'Outcome of Diabetes': str(result_diabetes), 'Outcome of Heart': str(result_heart), 'Outcome of Parkinsons': str(result_parkinson)})


if __name__ == '__main__':
    app.run(debug=True)
