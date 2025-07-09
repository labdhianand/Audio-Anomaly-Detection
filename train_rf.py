import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib

from dataset import train_data, test_data  #import data from datsset.py file

# flattening the spectrograms to 1d feature vectors
def extract_features(dataset):
    X, y = [], []
    for mel_db, label in dataset:
        feature = mel_db.numpy().flatten()  # shape:(1, 64, 64) -> (4096,)
        X.append(feature)
        y.append(label.item())  #convert to scalar
    return np.array(X), np.array(y)


# extractign features from train and test sets
X_train, y_train = extract_features(train_data)
X_test, y_test = extract_features(test_data)

# enable grid search to test and evaluate best combo of hyper parameters
grid_search = True

if grid_search:
    param_grid = {
        'n_estimators': [25, 50, 100],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt'],
        'class_weight': ['balanced']
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        scoring='f1',
        cv=5,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    clf = grid.best_estimator_
    print("Best Parameters:", grid.best_params_)

else:
    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42
    )
    clf.fit(X_train, y_train)


# predicting probability on test data
y_pred_proba = clf.predict_proba(X_test)[:, 1]  # probability for class 1 - anomaly
y_pred = clf.predict(X_test)

print("\nAnomaly Scores for Test Files:")
for i, score in enumerate(y_pred_proba[:5]):
    label_str = "Anomalous" if y_test[i] == 1 else "Normal"
    print(f"Test file {i+1}: Score = {score:.4f} | Actual: {label_str}")

print("\nAccuracy:", accuracy_score(y_test, y_pred))


# save the trained model
joblib.dump(clf, "rf_model.pkl")
print("\nTrained model saved to: rf_model.pkl")

