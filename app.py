import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained linear regression model
with open("linear_model.pkl", "rb") as f:
    model = pickle.load(f)

# HTML Template with inline CSS design
HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Performance Predictor</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: #ffffff;
            padding: 30px 40px;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
            width: 100%;
            max-width: 480px;
        }
        h2 {
            text-align: center;
            color: #333;
            margin-bottom: 24px;
        }
        .form-group {
            margin-bottom: 18px;
        }
        label {
            display: block;
            margin-bottom: 6px;
            color: #555;
            font-weight: 600;
            font-size: 14px;
        }
        input[type="number"] {
            width: 100%;
            padding: 10px 14px;
            border: 1px solid #ccc;
            border-radius: 6px;
            font-size: 15px;
            transition: border-color 0.2s;
        }
        input[type="number"]:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            width: 100%;
            padding: 12px;
            background-color: #667eea;
            color: #ffffff;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: background-color 0.2s;
            margin-top: 10px;
        }
        button:hover {
            background-color: #5a67d8;
        }
        .result {
            margin-top: 25px;
            padding: 15px;
            background-color: #e3f2fd;
            border-left: 5px solid #2196f3;
            border-radius: 4px;
            text-align: center;
        }
        .result h3 {
            color: #0d47a1;
            font-size: 18px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Performance Predictor</h2>
        <form action="/predict" method="POST">
            <div class="form-group">
                <label for="hours_studied">Hours Studied</label>
                <input type="number" step="any" id="hours_studied" name="hours_studied" required placeholder="e.g. 7">
            </div>
            <div class="form-group">
                <label for="extracurricular">Extracurricular Activities (Hours)</label>
                <input type="number" step="any" id="extracurricular" name="extracurricular" required placeholder="e.g. 3">
            </div>
            <div class="form-group">
                <label for="sleep_hours">Sleep Hours</label>
                <input type="number" step="any" id="sleep_hours" name="sleep_hours" required placeholder="e.g. 8">
            </div>
            <div class="form-group">
                <label for="question_papers">Sample Question Papers Practiced</label>
                <input type="number" step="any" id="question_papers" name="question_papers" required placeholder="e.g. 5">
            </div>
            <button type="submit">Predict Score</button>
        </form>

        {% if prediction_text %}
        <div class="result">
            <h3>Predicted Result: {{ prediction_text }}</h3>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_LAYOUT)

@app.route("/predict", methods=["POST"])
def predict():
    # Extract numerical features in the exact input sequence expected by the pickle file
    features = [
        float(request.form["hours_studied"]),
        float(request.form["extracurricular"]),
        float(request.form["sleep_hours"]),
        float(request.form["question_papers"])
    ]
    
    # Predict score using model
    final_features = np.array([features])
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)

    return render_template_string(HTML_LAYOUT, prediction_text=f"{output}")

if __name__ == "__main__":
    app.run(debug=True)
