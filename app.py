from flask import Flask, request, render_template
import pickle
import numpy as np

# Load your trained insurance price model
with open("Insurence_price_model (1).pkl", "rb") as file:
    model = pickle.load(file)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Extract numeric features from form
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    # Predict insurance price
    prediction = model.predict(final_features)
    output = round(prediction[0]/10, 2)  # round to 2 decimal places
    
    return render_template("index.html", prediction_text=f"Predicted Insurance Price: ₹{output}")

if __name__ == "__main__":
    app.run(debug=True)
