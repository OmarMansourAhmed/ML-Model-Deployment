import numpy as np
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the pickle model
try:
    model = pickle.load(open('_model_rf.pkl', 'rb'))
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form values
        feature_keys = ['lead_time', 'average_price', 'special_requests', 'Month',
       'number_of_week_nights', 'number_of_weekend_nights',
       'market_segment_type_Online', 'number_of_adults',
       'market_segment_type_Offline', 'type_of_meal_Not Selected',
       'room_type_Room_Type 4', 'type_of_meal_Meal Plan 2']
        float_features = [float(request.form[key]) for key in feature_keys]
        features = [np.array(float_features)]

        # Check if models are loaded
        if model:
            prediction1 = model.predict(features)
            output1 = prediction1[0]  # Assuming the prediction is a single value array
            return render_template('index.html', prediction_text1=f'RF Prediction [Not Canceled-> 0 , Canceled-> 1]: {output1}')
        else:
            return render_template('index.html', prediction_text1='Model 1 not loaded')
    except Exception as e:
        return render_template('index.html', prediction_text1=f'Error in Model 1 prediction: {e}')

if __name__ == '__main__':
    app.run(debug=True)
