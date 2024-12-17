from flask import Flask, request, jsonify
from flask_cors import CORS  
import pickle  
import pandas as pd  

app = Flask(__name__)
CORS(app)  


with open('naive_bayes_model_bmicase.pkl', 'rb') as model_file:
    knn_model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

def load_exercise_plans(csv_file):
    return pd.read_csv(csv_file, encoding='ISO-8859-1')

@app.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    if request.is_json:
        data = request.get_json()
        gender = data.get('gender')
        age = data.get('age')
        weight = float(data.get('weight'))  
        height = float(data.get('height'))

        bmi = weight / (height ** 2)

        features = [weight, height, bmi, age]  
        if gender.lower() == 'male':
            features.append(1)  
        else:
            features.append(2)  

        features_scaled = scaler.transform([features])

        bmi_case = knn_model.predict(features_scaled)[0]

        exercise_plans = load_exercise_plans('workout_plan.csv')

        plan_row = exercise_plans[exercise_plans['BMIcase'].str.lower() == bmi_case.lower()]

        if not plan_row.empty:
            workout_plan = {
                "Exercise Recommendation Plan": str(plan_row['Exercise Recommendation Plan'].values[0])
            }
        else:
            workout_plan = {
                "Exercise Recommendation Plan": "No workout plan found for your BMI case.",
            }

        recommendation = {
            "BMI": bmi,
            "BMIcase": bmi_case,
            "RecommendedExercisePlan": workout_plan
        }

        return jsonify(recommendation)
    
    else:
        return jsonify({"error": "Invalid content type, expected application/json"}), 415

if __name__ == '__main__':
    app.run(debug=True)
