import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle

def load_and_preprocess(filepath):
    data = pd.read_csv(filepath)

    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 2})

    features = data[['Weight', 'Height', 'BMI', 'Gender', 'Age']]
    target_bmicase = data['BMIcase'] 
    target_exercise = data['Exercise Recommendation Plan'] 

    return features, target_bmicase, target_exercise

def train_knn_model(features, target_bmicase, target_exercise, n_neighbors=3):
    X_train, X_test, y_train_bmicase, y_test_bmicase, y_train_exercise, y_test_exercise = train_test_split(
        features, target_bmicase, target_exercise, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn_model_bmicase = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model_bmicase.fit(X_train_scaled, y_train_bmicase)

    knn_model_exercise = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model_exercise.fit(X_train_scaled, y_train_exercise)

    with open('knn_model_bmicase.pkl', 'wb') as model_file:
        pickle.dump(knn_model_bmicase, model_file)

    with open('knn_model_exercise.pkl', 'wb') as model_file:
        pickle.dump(knn_model_exercise, model_file)

    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    return knn_model_bmicase, knn_model_exercise, scaler

if __name__ == "__main__":
    features, target_bmicase, target_exercise = load_and_preprocess('fitness_dataset.csv')
    train_knn_model(features, target_bmicase, target_exercise)
    print("Model and scaler saved successfully!")
