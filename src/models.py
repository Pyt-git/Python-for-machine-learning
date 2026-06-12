from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def split_data(x, y): 
    return train_test_split(
        X, y test_size = 0.2, random_state = 42, stratify = y
    )

def train_classifier(X_train, y_train): 
    clf = LogisticRegression(max_iter = 200)
    clf.fit(X_train, y_train)
    return clf

def evaluate(clf, X_test, y_test): 
    y_pred = clf.predict(X_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
