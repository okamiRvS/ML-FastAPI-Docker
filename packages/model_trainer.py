from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

def run_model_training(X_train, X_test, y_train, y_test):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    print(classification_report(y_test, y_pred))

    return gnb