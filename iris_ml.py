from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


MODEL_PATH = Path(__file__).with_name("iris_model.pkl")


def train_and_save_model():
    iris = load_iris(as_frame=True)
    features = iris.data
    target = iris.target

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    bundle = {
        "model": model,
        "feature_names": list(iris.feature_names),
        "target_names": list(iris.target_names),
        "accuracy": accuracy_score(y_test, predictions),
        "report": classification_report(
            y_test,
            predictions,
            target_names=iris.target_names,
            output_dict=True,
        ),
        "training_samples": len(x_train),
    }

    joblib.dump(bundle, MODEL_PATH)
    return bundle


if __name__ == "__main__":
    saved_bundle = train_and_save_model()
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Validation accuracy: {saved_bundle['accuracy']:.1%}")
