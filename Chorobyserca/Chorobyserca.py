import pandas as pd
import time
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Load dataset
file_path = "C:/Users/zuzan/Desktop/PRO1D/processed.cleveland.data"
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
    "num"
]

data = pd.read_csv(file_path, header=None, names=columns, na_values="?")

# Przeglądanie danych
print("Data Info:\n")
data.info()
print("\nPierwsze piec rzedow danych:\n")
print(data.head())
print(data.shape)

# Usuwanie brakujacych wartosci
print("\nBrakujace wartosci:\n")
print(data.isnull().sum())
data_cleaned = data.dropna()

# data shape
print(f"\nDane po usunieciu brakujacych wartosci: {data_cleaned.shape}")
print("\nLiczba diagnoz (num target kolumna):\n")
print(data_cleaned["num"].value_counts())

correlation_matrix = data_cleaned.corr()

# Wybieranie cech
correlation_threshold = 0.4
significant_features = correlation_matrix.index[abs(correlation_matrix["num"]) > correlation_threshold].tolist()
significant_features.remove("num")

print("\nWybrane cechy:\n")
print(significant_features)

X = data_cleaned[significant_features]
y = data_cleaned["num"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print(f"\nDane treningowe: {X_train.shape}, Dane testowe: {X_test.shape}")

# Normalizacja danych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ewaluacja klasyfikatorow
def evaluate_classifier(clf, param_grid, name):
    print(f"\nEvaluating {name}...")
    start_time = time.time()
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='f1_macro', verbose=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    predictions = best_model.predict(X_test)
    f1 = f1_score(y_test, predictions, average='macro')
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions)
    end_time = time.time()
    print(f"{name} completed in {end_time - start_time:.2f} seconds")
    print(f"Best Parameters: {best_params}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)


# Neural Network
nn_params = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['tanh', 'relu'],
    'solver': ['adam', 'sgd']
}
evaluate_classifier(MLPClassifier(max_iter=1000), nn_params, "Neural Network")

# Naive Bayes
nb_params = {}
evaluate_classifier(GaussianNB(), nb_params, "Naive Bayes")

# SVM
svm_params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'class_weight': ['balanced']
}
evaluate_classifier(SVC(probability=True), svm_params, "Support Vector Machine")

# Decision Tree
dt_params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
evaluate_classifier(DecisionTreeClassifier(), dt_params, "Decision Tree")

# Bagging
bagging_params = {
    'n_estimators': [10, 50, 100],
    'max_samples': [0.5, 1.0],
    'max_features': [0.5, 1.0]
}
evaluate_classifier(BaggingClassifier(), bagging_params, "Bagging")

# Boosting
boosting_params = {
    'n_estimators': [10, 50, 100],
    'learning_rate': [0.01, 0.1, 1.0]
}
evaluate_classifier(AdaBoostClassifier(), boosting_params, "Boosting")

# Reguly decyzyjne
rule_based_params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
evaluate_classifier(DecisionTreeClassifier(), rule_based_params, "Rule-based Classifier")
