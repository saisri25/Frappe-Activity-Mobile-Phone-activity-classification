import os
import pickle
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)
import joblib
import pickle
# Load the model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Load the encoders
encoders_path = os.path.dirname(os.path.abspath(__file__))

dayencoder = joblib.load(os.path.join(encoders_path, 'DayTimeEncoder.pkl'))
wkencoder = joblib.load(os.path.join(encoders_path, 'WeekdayEncoder.pkl'))
wkndencoder = joblib.load(os.path.join(encoders_path, 'WkndEncoder.pkl'))
hwencoder = joblib.load(os.path.join(encoders_path, 'hwencoder.pkl'))
wencoder = joblib.load(os.path.join(encoders_path, 'WeatherEncoder.pkl'))
cencoder = joblib.load(os.path.join(encoders_path, 'CostEncoder.pkl'))
#nencoder = joblib.load(os.path.join(encoders_path, 'NameEncoder.pkl'))

# Route to home page
@app.route('/')
def home():
    return render_template("home.html")

# Route to predict page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    return render_template("predict.html")

# Route to handle prediction
@app.route('/predictionpage', methods=['POST'])
def predictionpage():
    try:
        # Read CSV file for additional data if needed
        df = pd.read_csv(os.path.join(encoders_path, "frappe.csv"))

        # Extract features from the form
        item = int(request.form["item"])  # Assuming item is numeric
        daytime = dayencoder.transform([request.form["daytime"].lower()])[0]
        weekday = wkencoder.transform([request.form["weekday"].lower()])[0]
        
        # Determine if it's a weekend or weekday based on the day of the week
        if request.form["weekday"].lower() in ['sunday', 'saturday']:
            iswknd = 1  # weekend
        else:
            iswknd = 0  # weekday
        
        cost = cencoder.transform([request.form["cost"].lower()])[0]
        weather = wencoder.transform([request.form["weather"].lower()])[0]
        city = int(request.form["city"])  # Assuming city is numeric
        #name=nencoder.transform(([request.form["name"].lower()])[0])
        # Prepare input for prediction
        x_test = [[item, daytime, weekday, iswknd, cost, weather, city]]
        
        # Scale the input
        x_test = scaler.transform(x_test)

        # Make prediction
        pred = model.predict(x_test)

        # Interpret prediction result
        if pred[0] == 0:
            result = "Homework"
        elif pred[0] == 1:
            result = "Unknown"
        else:
            result = "Work"

        prediction_text = f"The phone activity was most likely for {result}"
    
    except KeyError as e:
        return render_template("predict.html", error_message=f"KeyError: {str(e)}. Please check your input.")

    except Exception as e:
        return render_template("predict.html", error_message=str(e))

    return render_template("predictionpage.html", prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
