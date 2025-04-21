from flask import Flask, request, jsonify, render_template, redirect, session
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import torch
import os
import logging
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from backend.main import DeepNeuralNetwork

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)
app.secret_key = 'your_secret_key'

# Global model variables
classical_model = None
dl_model = None
scaler = None
label_encoders = None

# ======================
# 🛠️ Init Database
# ======================
# Add this code somewhere before any database operations to reset the database


# Create the database and users table
def init_db():
    db_path = 'users.db'
    if not os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT NOT NULL,
                            email TEXT NOT NULL UNIQUE,
                            password TEXT NOT NULL
                        )''')
        conn.commit()
        conn.close()


# Call init_db() to reset the table when needed



# ======================
# 🔐 Signup
# ======================
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])

        try:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", (name, email, password))
            conn.commit()
            conn.close()
            return redirect('/login')
        except sqlite3.IntegrityError:
            return render_template('signup.html', error='Email already exists.')
    
    return render_template('signup.html')

# ======================
# 🔓 Login
# ======================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute("SELECT password, name FROM users WHERE email = ?", (email,))
        result = cursor.fetchone()
        conn.close()

        if result and check_password_hash(result[0], password):
            session['email'] = email
            session['name'] = result[1]
            return redirect('/')
        else:
            return render_template('login.html', error='Invalid credentials.')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('email', None)
    session.pop('name', None)
    return redirect('/login')

# ======================
# 📦 Load Models
# ======================
def load_models():
    global classical_model, dl_model, scaler, label_encoders
    try:
        with open('backend/preprocessing.pkl', 'rb') as f:
            preprocessing = pickle.load(f)
            scaler = preprocessing['scaler']
            label_encoders = preprocessing['label_encoders']
        with open('backend/classical_model.pkl', 'rb') as f:
            classical_model = pickle.load(f)
        n_classes = len(label_encoders['Treatment'].classes_)
        dl_model = DeepNeuralNetwork(input_size=6, hidden_size=64, output_size=n_classes)
        dl_model.load_state_dict(torch.load('backend/dl_model.pth'))
        dl_model.eval()
        return True
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        return False

load_models()

# ======================
# 🏠 Home
# ======================
@app.route('/')
def home():
    if 'email' not in session:
        return redirect('/login')
    return render_template('index.html', name=session['name'])

# ======================
# 🤖 Predict
# ======================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        required_fields = ['Age', 'Sex', 'Grade', 'HistologicalType', 'MSKCCType', 'SiteOfPrimarySTS']
        
        # Check if all required fields are present
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'Missing or empty value for {field}'}), 400

        # Prepare data for prediction
        input_df = pd.DataFrame([{
            'Age': float(data['Age']),
            'Sex': data['Sex'],
            'Grade': data['Grade'],
            'Histological type': data['HistologicalType'],
            'MSKCC type': data['MSKCCType'],
            'Site of primary STS': data['SiteOfPrimarySTS']
        }])

        # Check if all input values are valid
        for col in input_df.columns:
            if col in label_encoders:
                possible_values = list(label_encoders[col].classes_)
                if input_df[col].iloc[0] not in possible_values:
                    return jsonify({'error': f'Invalid value for {col}. Valid values: {", ".join(possible_values)}'}), 400
                input_df[col] = label_encoders[col].transform(input_df[col])

        # Scale the input data
        input_scaled = scaler.transform(input_df.values)

        # Check input shape and reshape if necessary
        input_tensor = torch.FloatTensor(input_scaled)

        # If the model expects a 2D tensor (batch_size, num_features)
        if input_tensor.ndimension() == 1:
            input_tensor = input_tensor.unsqueeze(0)

        # Classical model prediction
        classical_pred = classical_model.predict(input_scaled)[0]
        classical_proba = classical_model.predict_proba(input_scaled)[0]

        # Deep learning model prediction
        with torch.no_grad():
            dl_output = dl_model(input_tensor)
            dl_pred = torch.argmax(dl_output, dim=1).item()
            dl_proba = torch.softmax(dl_output, dim=1).numpy()[0]

        # Inverse transform for predicted labels
        treatment_encoder = label_encoders['Treatment']
        classical_label = treatment_encoder.inverse_transform([classical_pred])[0]
        dl_label = treatment_encoder.inverse_transform([dl_pred])[0]

        return jsonify({
            'input_data': data,
            'recommendations': {
                'classical_ml': classical_label,
                'deep_learning': dl_label
            },
            'all_treatments': list(treatment_encoder.classes_),
            'treatment_probabilities': {
                'classical_ml': classical_proba.tolist(),
                'deep_learning': dl_proba.tolist()
            }
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ======================
# ▶️ Run App
# ======================
if __name__ == '__main__':
    init_db()
    app.run(debug=True)
