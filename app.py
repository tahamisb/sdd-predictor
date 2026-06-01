from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import json
import pandas as pd
import requests as req

app = Flask(__name__)
CORS(app)

model = pickle.load(open('sdd_model.pkl', 'rb'))
features = json.load(open('features.json', 'r'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/fetch-latest', methods=['GET'])
def fetch_latest():
    try:
        response = req.get('https://horde.thunderstrikemarket.org/item-history?item_id=6657&faction=horde')
        data = response.json()
        latest = data['hourlyPoints'][-1]
        return jsonify(latest)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    for key in data:
        data[key] = float(data[key])
    df = pd.DataFrame([data])
    prediction = float(model.predict(df[features])[0])
    current_price = float(data['price'])
    should_post = current_price >= prediction * 0.95
    return jsonify({
        'predicted_price': round(float(prediction/100), 1),
        'current_price': round(float(current_price/100), 1),
        'recommendation': 'POST NOW' if should_post else 'WAIT',
        'difference': round(float(current_price - prediction)/100, 1)
    })

if __name__ == '__main__':
    app.run(debug=True)