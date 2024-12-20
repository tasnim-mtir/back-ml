import pandas as pd
import pickle

def load_exercise_plans(csv_file):
    return pd.read_csv(csv_file, encoding='ISO-8859-1')

def load_model_and_scaler(model_bmicase_path, model_exercise_path, scaler_path):
    with open(model_bmicase_path, 'rb') as model_file:
        svm_model_bmicase2 = pickle.load(model_file)
    
    with open(model_exercise_path, 'rb') as model_file:
        svm_model_exercise2 = pickle.load(model_file)
    
    with open(scaler_path, 'rb') as scaler_file:
        scaler2 = pickle.load(scaler_file)
    
    return svm_model_bmicase2, svm_model_exercise2, scaler2
def recommend_workout(features, model_bmicase, model_exercise, scaler):
    feature_columns = ['Weight', 'Height', 'BMI', 'Gender', 'Age']
    features_df = pd.DataFrame([features], columns=feature_columns)
    
    features_scaled = scaler.transform(features_df)
    
    bmicase = model_bmicase.predict(features_scaled)[0]
    exercise_recommendation = model_exercise.predict(features_scaled)[0]
    
    return bmicase, exercise_recommendation

def chatbot():
    print("Welcome to the Fitness Chatbot!")
    
    gender = input("Please enter your gender (Male/Female): ")
    age = int(input("Please enter your age: "))
    weight = float(input("Please enter your weight in kg: "))
    height = float(input("Please enter your height in meters (e.g., 1.70): "))
    
    bmi = weight / (height ** 2)
    print(f"Your BMI is: {bmi:.2f}")
    
    gender_numeric = 1 if gender.lower() == 'male' else 2  
    features = [weight, height, bmi, gender_numeric, age]
    
    model_bmicase, model_exercise, scaler = load_model_and_scaler(
        'svm_model_bmicase2.pkl', 'svm_model_exercise2.pkl', 'scaler2.pkl')
    
    bmicase, recommended_plan = recommend_workout(features, model_bmicase, model_exercise, scaler)
    
    print(f"BMI Case: {bmicase}")
    
    exercise_plans = load_exercise_plans('workout_plan.csv')
    
    plan_row = exercise_plans[exercise_plans['BMIcase'].str.lower() == bmicase.lower()]
    
    if not plan_row.empty:
        print(f"Goal: {plan_row['Goal'].values[0]}")
        print(f"Warm-up: {plan_row['Warm-up'].values[0]}")
        print(f"Strength Training: {plan_row['Strength Training'].values[0]}")
        print(f"Cardio: {plan_row['Cardio'].values[0]}")
        print(f"Cool Down: {plan_row['Cool Down'].values[0]}")
    else:
        print("No workout plan found for your BMI case.")

if __name__ == "__main__":
    chatbot()
