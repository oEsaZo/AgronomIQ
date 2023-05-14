from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the pre-trained model
with open('model1.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/yeild_predict', methods=['POST'])
def predict():
    data = request.get_json()

    state = float(data['state'])
    district = float(data['district'])
    year = float(data['year'])
    season = float(data['season'])
    crop = float(data['crop'])
    temperature = float(data['temperature'])
    humidity = float(data['humidity'])
    soil_moisture = float(data['soilmoisture'])
    area = float(data['area'])

    # Preprocess the input data (if necessary)
    # ...

    # Create input array
    input_data = np.array([[state, district, year, season, crop, temperature, humidity, soil_moisture, area]])

    # Make prediction
    prediction = model.predict(input_data)

    # Format the prediction result
    result = {'prediction': float(prediction[0])}

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
