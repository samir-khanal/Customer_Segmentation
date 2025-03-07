from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from model import load_model, predict_segment
from model import preprocess_input

# Initialize the Flask application
app = Flask(__name__)

# Loading the trained model and scaler
model = load_model()

# Home Route (User-friendly page)
@app.route('/')
def home():
    return render_template('index.html')  # Loads an HTML file for the homepage

# Prediction Route(API Endpoint)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # get the input data from the request (in JSON format)
        data = request.json
        # converting input data into a Pandas DataFrame (needed for preprocessing)
        df = pd.DataFrame([data])
        # preprocess the input data 
        processed_data = preprocess_input(df)
        # predicting the customer segment using the trained model
        prediction = predict_segment(model, processed_data)
        # Returns the predicted cluster as a JSON response
        return jsonify({'cluster': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
