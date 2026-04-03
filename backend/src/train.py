from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import joblib
import os


def train_model():
    iris = load_iris()
    X, y = iris.data, iris.target

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)

    # Save model into the model/ folder (one level above src)
    os.makedirs("../model", exist_ok=True)
    joblib.dump(model, "../model/iris_model.pkl")
    print("Model trained and saved successfully at ../model/iris_model.pkl")


if __name__ == "__main__":
    train_model()
