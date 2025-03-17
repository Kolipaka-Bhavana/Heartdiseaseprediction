from flask import Flask, render_template, request, jsonify, g
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from flask import Flask, send_from_directory

app = Flask(__name__)

app = Flask(__name__)

# Load dataset
DATA_FILE = "heart.csv"
df = pd.read_csv(DATA_FILE)

# Define categorical and numerical columns
categorical_columns = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
numerical_columns = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

# Encode categorical features
encoders = {col: LabelEncoder() for col in categorical_columns}
for col in categorical_columns:
    df[col] = encoders[col].fit_transform(df[col])

# Fit a scaler for numerical values
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Lazy model loading
MODEL_FILE = "my_model.h5"

def get_model():
    """Load the CNN model only once per request for efficiency."""
    if "model" not in g:
        g.model = tf.keras.models.load_model(MODEL_FILE)
    return g.model

def preprocess_input(input_data):
    """
    Convert input dictionary to a properly formatted NumPy array for the CNN model.
    """
    try:
        # Convert numerical inputs to float and scale
        for col in numerical_columns:
            input_data[col] = float(input_data[col])
        
        input_df = pd.DataFrame([input_data])
        input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

        # Encode categorical features with a fallback for unseen values
        for col in categorical_columns:
            if input_data[col] in encoders[col].classes_:
                input_df[col] = encoders[col].transform([input_data[col]])[0]
            else:
                input_df[col] = 0  # Default to the first category to avoid errors

        # Reshape for CNN input
        processed_array = input_df.values.reshape(1, -1, 1)
        return processed_array, None

    except Exception as e:
        return None, str(e)

# Home Route
@app.route('/')
def home():
    return render_template('home.html')

# About Page
@app.route('/about')
def about():
    return render_template('about.html')

# Metrics Page
@app.route('/metrics')
def metrics():
    return render_template('metrics.html')

# Flowchart Page
@app.route('/flowchart')
def flowchart():
    return render_template('flowchart.html')

# Prediction Page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')

    if request.method == 'POST':
        try:
            input_data = request.form.to_dict()

            # Preprocess input
            processed_input, error = preprocess_input(input_data)
            if error:
                return render_template('predict.html', error=error)

            # Make prediction
            prediction = get_model().predict(processed_input)
            result = "Heart Disease Detected" if prediction[0][0] > 0.5 else "No Heart Disease"

            return render_template('predict.html', prediction=result)  # Show result on UI
        
        except Exception as e:
            return render_template('predict.html', error=str(e))

# API for CRUD operations on dataset
@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify(df.to_dict(orient='records'))

@app.route('/api/data', methods=['POST'])
def add_data():
    try:
        new_data = request.get_json()
        df.loc[len(df)] = new_data
        df.to_csv(DATA_FILE, index=False)
        return jsonify({'message': 'Data added successfully'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/data/<int:index>', methods=['PUT'])
def update_data(index):
    try:
        updated_data = request.get_json()
        if index >= len(df):
            return jsonify({'error': 'Index out of range'}), 404
        df.loc[index] = updated_data
        df.to_csv(DATA_FILE, index=False)
        return jsonify({'message': 'Data updated successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/data/<int:index>', methods=['DELETE'])
def delete_data(index):
    try:
        if index >= len(df):
            return jsonify({'error': 'Index out of range'}), 404
        df.drop(index, inplace=True)
        df.reset_index(drop=True, inplace=True)  # Reset index after deletion
        df.to_csv(DATA_FILE, index=False)
        return jsonify({'message': 'Data deleted successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico', mimetype='image/vnd.microsoft.icon')


# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
