import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')





@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')


@app.route('/diabetes/predict', methods=['POST'])
def diabetes_predict():
    try:
        diabetes_pipe = pickle.load(open("models/diabetes_model.pkl", 'rb'))

        Pregnancies = request.form.get('Pregnancies')
        Glucose = request.form.get('Glucose')
        BloodPressure = request.form.get('BloodPressure')
        BMI = request.form.get('BMI')
        DiabetesPedigreeFunction = request.form.get('DiabetesPedigreeFunction')
        Age = request.form.get('Age')

        input_data = (
        int(Pregnancies), float(Glucose), float(BloodPressure), float(BMI), float(DiabetesPedigreeFunction), int(Age))
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        prediction = diabetes_pipe.predict(input_data_reshaped)
        print(diabetes_pipe.predict_proba(input_data_reshaped))
        prob_predict = diabetes_pipe.predict_proba(input_data_reshaped)[0][1]*100
        print(prob_predict)

        return render_template('diabetes.html',error="Chance of having Diabetes: {0}%".format(str(round(prob_predict, 2))))

    except:
        return render_template('diabetes.html', error="Enter appropriate values")

@app.route('/heart')
def heart():

    return render_template('heart.html')

@app.route('/heart/predict', methods=['POST'])
def heart_predict():
    try:
        heart_data = pd.read_csv('datasets/heart_disease_data.csv')
        heart_pipe = pickle.load(open("models/heart_disease_model.pkl", 'rb'))
        sc = StandardScaler()
        X = heart_data[['cp', 'thalach', 'exang', 'oldpeak', 'ca', 'thal']].values
        sc.fit(X)

        cp = request.form.get('cp')
        thalach = request.form.get('thalach')
        exang = request.form.get('exang')
        oldpeak = request.form.get('oldpeak')
        ca = request.form.get('ca')
        thal = request.form.get('thal')

        input_data = (
            int(cp), int(thalach), int(exang),
            float(oldpeak), int(ca), int(thal))
        print(input_data)
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        std_data = sc.transform(input_data_reshaped)
        prediction = heart_pipe.predict(std_data)
        prob_predict = heart_pipe.predict_proba(std_data)[0][1] * 100

        print(prob_predict)

        return render_template('heart.html',
                               error="Chance of having Heart Disease: " + str(round(prob_predict, 2)) + "%")


    except:
        return render_template('heart.html', error="Enter appropriate values")


@app.route('/liver')
def liver():
    return render_template('liver.html')

@app.route('/liver/predict', methods=['POST'])
def liver_predict():
    try:
        liver_pipe = pickle.load(open("models/liver_disease_model.pkl", 'rb'))

        Age = request.form.get('Age')
        Total_Bilirubin = request.form.get('Total_Bilirubin')
        Direct_Bilirubin = request.form.get('Direct_Bilirubin')
        Alkaline_Phosphotase = request.form.get('Alkaline_Phosphotase')
        Alamine_Aminotransferase = request.form.get('Alamine_Aminotransferase')
        Aspartate_Aminotransferase = request.form.get('Aspartate_Aminotransferase')
        Albumin = request.form.get('Albumin')
        Albumin_and_Globulin_Ratio = request.form.get('Albumin_and_Globulin_Ratio')

        input_data = (
            int(Age), float(Total_Bilirubin), float(Direct_Bilirubin), int(Alkaline_Phosphotase),
            int(Alamine_Aminotransferase)
            , int(Aspartate_Aminotransferase), float(Albumin), float(Albumin_and_Globulin_Ratio))
        print(input_data)
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        prediction = liver_pipe.predict(input_data_reshaped)
        prob_predict = liver_pipe.predict_proba(input_data_reshaped)[0][1] * 100

        return render_template('liver.html',error="Chance of having Liver Disease: " + str(round(prob_predict, 2)) + "%")

    except:
        return render_template('liver.html', error="Enter appropriate values")


@app.route('/breastcancer')
def breastcancer():
    return render_template('breastcancer.html')

@app.route('/breastcancer/predict', methods=['POST'])
def breastcancer_predict():
    try:

        breastcancer_pipe = pickle.load(open("models/breast_cancer_model.pkl", 'rb'))

        radius_mean = request.form.get('radius_mean')
        perimeter_mean = request.form.get('perimeter_mean')
        area_mean = request.form.get('area_mean')
        concavity_mean = request.form.get('concavity_mean')
        concave_points_mean = request.form.get('concave_points_mean')
        radius_worst = request.form.get('radius_worst')
        perimeter_worst = request.form.get('perimeter_worst')
        area_worst = request.form.get('area_worst')
        concavity_worst = request.form.get('concavity_worst')
        concave_points_worst = request.form.get('concave_points_worst')

        input_data = (float(radius_mean),float(perimeter_mean), float(area_mean),float(concavity_mean), float(concave_points_mean), float(radius_worst), float(perimeter_worst), float(area_worst),float(concavity_worst), float(concave_points_worst))
        print(input_data)
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        prediction = breastcancer_pipe.predict(input_data_reshaped)
        print(prediction)
        print(breastcancer_pipe.predict_proba(input_data_reshaped)[0][0])
        prob_predict_m = breastcancer_pipe.predict_proba(input_data_reshaped)[0][0] * 100
        prob_predict_b = breastcancer_pipe.predict_proba(input_data_reshaped)[0][1] * 100
        if prediction[0] != 1:
            return render_template('breastcancer.html',error="Malignant")
        else:
            return render_template('breastcancer.html',error="Benign")

    except:
        return render_template('breastcancer.html', error="Enter appropriate values")

@app.route('/kidney')
def kidney():
    return render_template('kidney.html')

@app.route('/kidney/predict', methods=['POST'])
def kidney_predict():
    try:
        kidney_pipe = pickle.load(open("models/chronic_kidney_disease_model.pkl", 'rb'))

        Sg = request.form.get('Sg')
        Al = request.form.get('Al')
        Rbc = request.form.get('Rbc')
        Pc = request.form.get('Pc')
        Hemo = request.form.get('Hemo')
        Pcv = request.form.get('Pcv')
        Htn = request.form.get('Htn')
        Dm = request.form.get('Dm')
        Appet = request.form.get('Appet')
        pe = request.form.get('pe')


        input_data = (
            float(Sg), float(Al), float(Rbc),float(Pc),float(Hemo),float(Pcv), float(Htn), float(Dm),float(Appet),float(pe))
        print(input_data)
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        prediction = kidney_pipe.predict(input_data_reshaped)
        prob_predict = kidney_pipe.predict_proba(input_data_reshaped)[0][1] * 100

        print(prob_predict)

        return render_template('kidney.html',error="Chance of having Kidney Disease: " + str(round(prob_predict, 2)) + "%")

    except:
        return render_template('kidney.html', error="Enter appropriate values")

if __name__ == "__main__":
    app.run(debug=True, port=5001)
