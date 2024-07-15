import numpy as np
import joblib  
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)
try:
    model = joblib.load('RF.joblib') 
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/predict')
def predict():
    """Render the prediction page."""
    return render_template('predict.html')

@app.route('/submit', methods=["POST"])
def submit():
    """Handle form submission and make predictions."""
    try:
        # Reading the inputs given by the user
        input_feature = [x for x in request.form.values()]
        input_feature = [np.array(input_feature)]

        # Define column names
        names = ['tempreature', 'humidity', 'water_level', 'N', 'P', 'K',
        'Watering_plant_pump', 'Fan_actuator']

        # Create a DataFrame
        data = pd.DataFrame(input_feature, columns=names)

        print(data)

        # Predictions using the loaded model file
        prediction = model.predict(data)
        print(prediction)

        # Determine the result message based on the prediction value
        if prediction[0] == 0:
            result = "Water pump actuator OFF"
            prediction = "OFF"
        else:
            result = "Water pump actuator ON"
            prediction = "ON"
        
        return render_template("Result.html", result=result, prediction=prediction)
    except Exception as e:
        # Handle exceptions and print error for debugging
        print(f"Error during prediction: {e}")
        return render_template("Result.html", result="An error occurred. Please try again.")

if __name__ == "__main__":
    app.run(debug=True, port=4000)  # Running the app with debug enabled
