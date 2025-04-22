from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ✅ FIX: Load model without compiling (skip optimizer)
model = load_model(
    r"D:\new project titils and documentation pdf\AI-Powered Cryptocurrency Price Prediction\Cryptocurrency-Prediction-with-Artificial-Intelligence-main\model\lstm_model.h5",
    compile=False
)

# Load the scaler
scaler = joblib.load(
    r"D:\new project titils and documentation pdf\AI-Powered Cryptocurrency Price Prediction\Cryptocurrency-Prediction-with-Artificial-Intelligence-main\model\scaler.pkl"
)

# Columns used in training
cols = ['open', 'high', 'low', 'close', 'Volume XRP', 'Volume USDT']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        data = []
        for i in range(1, 11):
            row = []
            for col in cols:
                row.append(float(request.form[f"{col}_{i}"]))
            data.append(row)

        df_input = pd.DataFrame(data, columns=cols)

        # Scale data
        input_scaled = scaler.transform(df_input)
        input_seq = np.expand_dims(input_scaled, axis=0)

        # Predict
        pred_norm = model.predict(input_seq)[0][0]

        # Denormalize prediction
        close_min = scaler.data_min_[3]  # 'close' is 4th column (index 3)
        close_max = scaler.data_max_[3]
        pred_orig = pred_norm * (close_max - close_min) + close_min

        return render_template('index.html',
                               prediction_text=f"Predicted Closing Price: {pred_orig:.5f} USDT",
                               norm_pred=f"(Normalized: {pred_norm:.5f})")
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
