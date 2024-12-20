import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

def load_model_and_scaler(model_bmicase_path, model_exercise_path, scaler_path):
    with open(model_bmicase_path, 'rb') as model_file:
        knn_model_bmicase = pickle.load(model_file)
    
    with open(model_exercise_path, 'rb') as model_file:
        knn_model_exercise = pickle.load(model_file)
    
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    
    return knn_model_bmicase, knn_model_exercise, scaler

def recommend_workout(features, model_bmicase, model_exercise, scaler):
    feature_columns = ['Weight', 'Height', 'BMI', 'Gender', 'Age']
    features_df = pd.DataFrame([features], columns=feature_columns)
    
    features_scaled = scaler.transform(features_df)
    
    bmicase = model_bmicase.predict(features_scaled)[0]
    exercise_recommendation = model_exercise.predict(features_scaled)[0]
    
    return bmicase, exercise_recommendation
