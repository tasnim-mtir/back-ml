import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import pickle

def load_and_preprocess(filepath):
    data = pd.read_csv(filepath)

    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 2})

    features = data[['Weight', 'Height', 'BMI', 'Gender', 'Age']]
    target_bmicase = data['BMIcase']  
    target_exercise = data['Exercise Recommendation Plan']  

    return features, target_bmicase, target_exercise

def train_naive_bayes_model(features, target_bmicase, target_exercise):
    X_train, X_test, y_train_bmicase, y_test_bmicase, y_train_exercise, y_test_exercise = train_test_split(
        features, target_bmicase, target_exercise, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    nb_model_bmicase = GaussianNB()
    nb_model_bmicase.fit(X_train_scaled, y_train_bmicase)

    nb_model_exercise = GaussianNB()
    nb_model_exercise.fit(X_train_scaled, y_train_exercise)

    with open('naive_bayes_model_bmicase.pkl', 'wb') as model_file:
        pickle.dump(nb_model_bmicase, model_file)

    with open('naive_bayes_model_exercise.pkl', 'wb') as model_file:
        pickle.dump(nb_model_exercise, model_file)

    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    return nb_model_bmicase, nb_model_exercise, scaler

if __name__ == "__main__":
    features, target_bmicase, target_exercise = load_and_preprocess('fitness_dataset.csv')
    train_naive_bayes_model(features, target_bmicase, target_exercise)
    print("Model and scaler saved successfully!")
