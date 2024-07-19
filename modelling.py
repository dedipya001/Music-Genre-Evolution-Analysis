from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import joblib

def train_and_evaluate_model(X, y):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                               param_grid=param_grid,
                               cv=5,
                               n_jobs=-1,
                               verbose=2)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f"Best Parameters: {best_params}")

    # Train with best parameters
    best_clf = grid_search.best_estimator_
    y_pred = best_clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save the model
    joblib.dump(best_clf, 'random_forest_model.pkl')

# def evaluate_additional_classifiers(X, y):
#     from sklearn.svm import SVC
#     from sklearn.ensemble import GradientBoostingClassifier
#     from sklearn.metrics import accuracy_score

#     # SVM
#     svm_clf = SVC(kernel='linear', random_state=42)
#     svm_clf.fit(X_train, y_train)
#     y_pred_svm = svm_clf.predict(X_test)
#     print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))

#     # Gradient Boosting
#     gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
#     gb_clf.fit(X_train, y_train)
#     y_pred_gb = gb_clf.predict(X_test)
#     print("Gradient Boosting Classification Report:\n", classification_report(y_test, y_pred_gb))


# from sklearn.svm import SVC
# from sklearn.metrics import classification_report

# def evaluate_additional_classifiers(X_train, X_test, y_train, y_test):
#     # SVM Classifier
#     svm_clf = SVC()
#     svm_clf.fit(X_train, y_train)
#     y_pred = svm_clf.predict(X_test)
#     print("SVM Classifier Report:")
#     print(classification_report(y_test, y_pred))

import joblib
from sklearn.metrics import classification_report

# Load pre-trained RandomForest model
model_path = 'random_forest_model.pkl'
clf = joblib.load(model_path)

def evaluate_model(X_test, y_test):
    # Predict using the loaded model
    y_pred = clf.predict(X_test)
    print("Random Forest Classifier Report:")
    print(classification_report(y_test, y_pred))