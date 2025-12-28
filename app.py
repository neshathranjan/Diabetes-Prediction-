from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and scaler
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    print("Error: Model or Scaler file not found. Please train the model first.")
    exit()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        features = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['bloodpressure']),
            float(request.form['skinthickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['dpf']),
            float(request.form['age'])
        ]
        
        # Preprocess matching training
        features_array = np.array([features])
        
        # Scale the data
        scaled_features = scaler.transform(features_array)
        
        # Predict
        prediction = model.predict(scaled_features)
        probability = model.predict_proba(scaled_features)[0][1] # Probability of being diabetic
        
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        prob_percent = round(probability * 100, 2)
        
        return render_template('result.html', result=result, probability=prob_percent)
        
    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, port=5000)
