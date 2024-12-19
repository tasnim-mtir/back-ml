from flask import Flask, request, jsonify
from flask_cors import CORS  
import pickle  
import numpy as np  

app = Flask(__name__)
CORS(app)  

with open('naive_bayes_model_bmicase.pkl', 'rb') as bmicase_model_file:
    nb_model_bmicase = pickle.load(bmicase_model_file)

with open('naive_bayes_model_exercise.pkl', 'rb') as exercise_model_file:
    nb_model_exercise = pickle.load(exercise_model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    if request.is_json:
        try:
            data = request.get_json()
            gender = data.get('gender')
            age = int(data.get('age'))
            weight = float(data.get('weight'))
            height = float(data.get('height'))

            bmi = weight / (height ** 2)

            gender_numeric = 1 if gender.lower() == 'male' else 2

            features = np.array([[weight, height, bmi, gender_numeric, age]])

            features_scaled = scaler.transform(features)

            bmicase_prediction = nb_model_bmicase.predict(features_scaled)[0]
            exercise_plan_prediction = nb_model_exercise.predict(features_scaled)[0]

            recommendation = {
                "BMI": round(bmi, 2),
                "BMIcase": bmicase_prediction,
                "RecommendedExercisePlan": int(exercise_plan_prediction)
            }

            return jsonify(recommendation), 200

        except Exception as e:
            return jsonify({"error": f"Error processing request: {str(e)}"}), 400
    else:
        return jsonify({"error": "Invalid content type, expected application/json"}), 415

if __name__ == '__main__':
    app.run(debug=True)
