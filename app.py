from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the models
models = {
    'Naive Bayes': joblib.load('model_joblib_NB_heart'),
    'decision_tree': joblib.load('model_joblib_decision_tree_heart'),
    'random_forest': joblib.load('model_joblib_rf_heart'),
    'knn': joblib.load('model_joblib_knn_heart')
}


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            age = int(request.form['age'])
            gender = int(request.form['gender'])
            cp = int(request.form['cp'])
            trestbps = int(request.form['trestbps'])
            chol = int(request.form['chol'])
            fbs = int(request.form['fbs'])
            restecg = int(request.form['restecg'])
            thalach = int(request.form['thalach'])
            exang = int(request.form['exang'])
            oldpeak = float(request.form['oldpeak'])
            slope = int(request.form['slope'])
            ca = int(request.form['ca'])
            thal = int(request.form['thal'])
            model_choice = request.form['model_choice']

            features = np.array([[age, gender, cp, trestbps, chol, fbs,
                                restecg, thalach, exang, oldpeak, slope, ca, thal]])

            model = models[model_choice]
            prediction = model.predict(features)[0]

            if prediction == 0:
                result = "No Heart Disease"
                color = "green"
            else:
                result = "Possibility of Heart Disease"
                color = "red"

            return render_template('predict.html', prediction=result, color=color)
        except Exception as e:
            return render_template('predict.html', error=str(e))
    return render_template('predict.html')


if __name__ == '__main__':
    app.run(debug=True)
