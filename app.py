
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
print("Loading model...")
with open('regressor_model.pkl', 'rb') as f:
    model = pickle.load(f)
print("Model loaded successfully.")

# Verify the type of the loaded object
print("Type of model:", type(model))

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the features from the request
    data = request.json

    # Extract the features needed for prediction
    features = [data['bedrooms'], data['bathrooms'], data['sqft_living'], data['condition'], data['yr_built']] # Add more features as needed

    # Convert the features to numpy array and reshape
    features = np.array(features).reshape(1, -1)

    # Make prediction
    try:
        prediction = model.predict(features)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': 'Error during prediction. Please check the server logs for more details.'})

if __name__ == "__main__":
    app.run(debug=True)