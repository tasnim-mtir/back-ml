import pickle
import numpy as np

with open('naive_bayes_model_bmicase.pkl', 'rb') as model_file:
    nb_model_bmicase = pickle.load(model_file)

with open('naive_bayes_model_exercise.pkl', 'rb') as model_file:
    nb_model_exercise = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

def calculate_bmi(weight, height):
    return weight / (height ** 2)

def predict_bmicase_and_exercise(weight, height, gender, age):
    bmi = calculate_bmi(weight, height)

    features = np.array([[weight, height, bmi, gender, age]])

    features_scaled = scaler.transform(features)

    bmicase_prediction = nb_model_bmicase.predict(features_scaled)[0]
    exercise_plan_prediction = nb_model_exercise.predict(features_scaled)[0]

    return bmi, bmicase_prediction, exercise_plan_prediction

if __name__ == "__main__":
    gender = input("Enter gender (Male/Female): ")
    age = int(input("Enter age (years): "))
    weight = float(input("Enter weight (kg): "))
    height = float(input("Enter height (m): "))

    gender_numeric = 1 if gender.lower() == 'male' else 2

    bmi, bmicase, exercise_plan = predict_bmicase_and_exercise(weight, height, gender_numeric, age)

    print(f"\nCalculated BMI: {bmi:.2f}")
    print(f"Predicted BMI Case: {bmicase}")
    print(f"Recommended Exercise Plan Level: {exercise_plan}")
