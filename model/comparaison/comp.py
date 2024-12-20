import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_and_preprocess(filepath):
    data = pd.read_csv(filepath)

    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 2})

    features = data[['Weight', 'Height', 'BMI', 'Gender', 'Age']]
    target_bmicase = data['BMIcase']  
    target_exercise = data['Exercise Recommendation Plan']  

    return features, target_bmicase, target_exercise

def train_and_evaluate_models(features, target_bmicase, target_exercise):
    X_train, X_test, y_train_bmicase, y_test_bmicase, y_train_exercise, y_test_exercise = train_test_split(
        features, target_bmicase, target_exercise, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entraîner le modèle KNN
    knn_model_bmicase = KNeighborsClassifier(n_neighbors=3)
    knn_model_bmicase.fit(X_train_scaled, y_train_bmicase)

    knn_model_exercise = KNeighborsClassifier(n_neighbors=3)
    knn_model_exercise.fit(X_train_scaled, y_train_exercise)

    # Entraîner le modèle SVM
    svm_model_bmicase = SVC()
    svm_model_bmicase.fit(X_train_scaled, y_train_bmicase)

    svm_model_exercise = SVC()
    svm_model_exercise.fit(X_train_scaled, y_train_exercise)

    # Entraîner le modèle Naive Bayes
    nb_model_bmicase = GaussianNB()
    nb_model_bmicase.fit(X_train_scaled, y_train_bmicase)

    nb_model_exercise = GaussianNB()
    nb_model_exercise.fit(X_train_scaled, y_train_exercise)

    # Prédictions avec le modèle KNN
    knn_bmicase_pred = knn_model_bmicase.predict(X_test_scaled)
    knn_exercise_pred = knn_model_exercise.predict(X_test_scaled)

    # Prédictions avec le modèle SVM
    svm_bmicase_pred = svm_model_bmicase.predict(X_test_scaled)
    svm_exercise_pred = svm_model_exercise.predict(X_test_scaled)

    # Prédictions avec le modèle Naive Bayes
    nb_bmicase_pred = nb_model_bmicase.predict(X_test_scaled)
    nb_exercise_pred = nb_model_exercise.predict(X_test_scaled)

    # Évaluation des modèles avec la précision et le rapport de classification
    knn_bmicase_acc = accuracy_score(y_test_bmicase, knn_bmicase_pred)
    svm_bmicase_acc = accuracy_score(y_test_bmicase, svm_bmicase_pred)
    nb_bmicase_acc = accuracy_score(y_test_bmicase, nb_bmicase_pred)

    knn_exercise_acc = accuracy_score(y_test_exercise, knn_exercise_pred)
    svm_exercise_acc = accuracy_score(y_test_exercise, svm_exercise_pred)
    nb_exercise_acc = accuracy_score(y_test_exercise, nb_exercise_pred)

    print("KNN Model (BMI Case):")
    print("Accuracy:", knn_bmicase_acc)
    print("Classification Report:\n", classification_report(y_test_bmicase, knn_bmicase_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test_bmicase, knn_bmicase_pred))

    print("\nSVM Model (BMI Case):")
    print("Accuracy:", svm_bmicase_acc)
    print("Classification Report:\n", classification_report(y_test_bmicase, svm_bmicase_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test_bmicase, svm_bmicase_pred))

    print("\nNaive Bayes Model (BMI Case):")
    print("Accuracy:", nb_bmicase_acc)
    print("Classification Report:\n", classification_report(y_test_bmicase, nb_bmicase_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test_bmicase, nb_bmicase_pred))

    print("\nKNN Model (Exercise Recommendation):")
    print("Accuracy:", knn_exercise_acc)
    print("Classification Report:\n", classification_report(y_test_exercise, knn_exercise_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test_exercise, knn_exercise_pred))

    print("\nSVM Model (Exercise Recommendation):")
    print("Accuracy:", svm_exercise_acc)
    print("Classification Report:\n", classification_report(y_test_exercise, svm_exercise_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test_exercise, svm_exercise_pred))

    print("\nNaive Bayes Model (Exercise Recommendation):")
    print("Accuracy:", nb_exercise_acc)
    print("Classification Report:\n", classification_report(y_test_exercise, nb_exercise_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test_exercise, nb_exercise_pred))

    return knn_bmicase_acc, svm_bmicase_acc, nb_bmicase_acc, knn_exercise_acc, svm_exercise_acc, nb_exercise_acc

if __name__ == "__main__":
    features, target_bmicase, target_exercise = load_and_preprocess('fitness_dataset.csv')
    knn_bmicase_acc, svm_bmicase_acc, nb_bmicase_acc, knn_exercise_acc, svm_exercise_acc, nb_exercise_acc = train_and_evaluate_models(features, target_bmicase, target_exercise)

    print("\nComparison of Models:")
    print("BMI Case (KNN vs SVM vs Naive Bayes):", knn_bmicase_acc, "vs", svm_bmicase_acc, "vs", nb_bmicase_acc)
    print("Exercise Recommendation (KNN vs SVM vs Naive Bayes):", knn_exercise_acc, "vs", svm_exercise_acc, "vs", nb_exercise_acc)
