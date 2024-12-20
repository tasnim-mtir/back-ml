import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle

def load_and_preprocess(filepath):
    data = pd.read_csv(filepath)

    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 2})

    features = data[['Weight', 'Height', 'BMI', 'Gender', 'Age']]
    target_bmicase = data['BMIcase']  
    target_exercise = data['Exercise Recommendation Plan']  

    return features, target_bmicase, target_exercise

def train_svm_model(features, target_bmicase, target_exercise, kernel='linear'):
    X_train, X_test, y_train_bmicase, y_test_bmicase, y_train_exercise, y_test_exercise = train_test_split(
        features, target_bmicase, target_exercise, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm_model_bmicase = SVC(kernel=kernel)
    svm_model_bmicase.fit(X_train_scaled, y_train_bmicase)

    svm_model_exercise = SVC(kernel=kernel)
    svm_model_exercise.fit(X_train_scaled, y_train_exercise)

    with open('svm_model_bmicase2.pkl', 'wb') as model_file:
        pickle.dump(svm_model_bmicase, model_file)

    with open('svm_model_exercise2.pkl', 'wb') as model_file:
        pickle.dump(svm_model_exercise, model_file)

    with open('scaler2.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    return svm_model_bmicase, svm_model_exercise, scaler

if __name__ == "__main__":
    features, target_bmicase, target_exercise = load_and_preprocess('fitness_dataset.csv')
    train_svm_model(features, target_bmicase, target_exercise)
    print("Model and scaler saved successfully!")
