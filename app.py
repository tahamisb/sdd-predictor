from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import json
import pandas as pd
import numpy as np
import requests as req
import schedule
import threading
import time
import os
from xgboost import XGBRegressor

app = Flask(__name__)
CORS(app)

FEATURES = [
    'hour', 'day_of_week', 'marketAvg', 'quantity',
    'auctions', 'soldDay', 'days_since_phase', 'price_vs_avg',
    'price_1h_ago', 'price_3h_ago', 'price_6h_ago', 'price_12h_ago',
    'price_change_1h', 'price_change_6h',
    'rolling_mean_6h', 'rolling_mean_12h', 'rolling_std_6h'
]

def build_features(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    phase_drop_date = pd.Timestamp('2026-05-16', tz='UTC')
    df['days_since_phase'] = (df['timestamp'] - phase_drop_date).dt.total_seconds() / 86400
    df['days_since_phase'] = df['days_since_phase'].clip(lower=0)

    df['price_vs_avg'] = df['price'] / df['marketAvg']
    df['soldDay'] = df['soldDay'].fillna(0)

    df['price_1h_ago'] = df['price'].shift(1)
    df['price_3h_ago'] = df['price'].shift(3)
    df['price_6h_ago'] = df['price'].shift(6)
    df['price_12h_ago'] = df['price'].shift(12)

    df['price_change_1h'] = df['price'] - df['price_1h_ago']
    df['price_change_6h'] = df['price'] - df['price_6h_ago']

    df['rolling_mean_6h'] = df['price'].rolling(6).mean()
    df['rolling_mean_12h'] = df['price'].rolling(12).mean()
    df['rolling_std_6h'] = df['price'].rolling(6).std()

    df = df.dropna()
    return df

def fetch_raw_data():
    response = req.get('https://horde.thunderstrikemarket.org/item-history?item_id=6657&faction=horde')
    data = response.json()
    return pd.DataFrame(data['hourlyPoints'])

def retrain():
    print("Retraining model with latest data...")
    try:
        df = fetch_raw_data()
        df = build_features(df)

        # Only use last 2 days - most relevant recent market
        cutoff = df['timestamp'].max() - pd.Timedelta(days=2)
        df = df[df['timestamp'] >= cutoff]
        print(f"Training on {len(df)} rows from last 2 days")

        X = df[FEATURES]
        y = df['price']

        new_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        new_model.fit(X, y)

        pickle.dump(new_model, open('sdd_model.pkl', 'wb'))

        with open('features.json', 'w') as f:
            json.dump(FEATURES, f)

        global model
        model = new_model

        print(f"Retrained successfully on {len(df)} rows.")

    except Exception as e:
        print(f"Retraining failed: {e}")

# Load initial model or train from scratch
try:
    model = pickle.load(open('sdd_model.pkl', 'rb'))
    print("Loaded existing model")
except:
    print("No model found - training from scratch")
    model = None
    retrain()

# Start background scheduler
def run_scheduler():
    schedule.every(24).hours.do(retrain)
    while True:
        schedule.run_pending()
        time.sleep(60)

# Only start scheduler and retrain once - not on Flask reloader restart
if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or not os.environ.get('WERKZEUG_RUN_MAIN'):
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    retrain()

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
def predict():
    try:
        df = fetch_raw_data()
        df = build_features(df)

        latest = df.iloc[-1]

        user_price = float(request.json.get('price', latest['price']))

        row = pd.DataFrame([{
            'hour': latest['hour'],
            'day_of_week': latest['day_of_week'],
            'marketAvg': latest['marketAvg'],
            'quantity': latest['quantity'],
            'auctions': latest['auctions'],
            'soldDay': latest['soldDay'],
            'days_since_phase': latest['days_since_phase'],
            'price_vs_avg': user_price / latest['marketAvg'],
            'price_1h_ago': latest['price_1h_ago'],
            'price_3h_ago': latest['price_3h_ago'],
            'price_6h_ago': latest['price_6h_ago'],
            'price_12h_ago': latest['price_12h_ago'],
            'price_change_1h': latest['price_change_1h'],
            'price_change_6h': latest['price_change_6h'],
            'rolling_mean_6h': latest['rolling_mean_6h'],
            'rolling_mean_12h': latest['rolling_mean_12h'],
            'rolling_std_6h': latest['rolling_std_6h']
        }])

        prediction = float(model.predict(row)[0])
        current_price = float(user_price)
        should_post = current_price >= prediction * 0.95

        return jsonify({
            'predicted_price': round(prediction/100, 1),
            'current_price': round(current_price/100, 1),
            'recommendation': 'POST NOW' if should_post else 'WAIT',
            'difference': round((current_price - prediction)/100, 1),
            'price_trend': 'RISING 📈' if latest['price_change_6h'] > 0 else 'FALLING 📉',
            'volatility': round(float(latest['rolling_std_6h'])/100, 1)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain_endpoint():
    retrain()
    return jsonify({'status': 'retrained successfully'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)