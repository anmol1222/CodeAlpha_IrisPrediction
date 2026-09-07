from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).with_name("iris_model.pkl")


st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="🌸",
    layout="centered",
)


@st.cache_resource
def load_model_bundle():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("The Iris model file was not found.")
    st.code("python iris_ml.py")
    st.stop()


model_bundle = load_model_bundle()
model = model_bundle["model"]
feature_names = model_bundle["feature_names"]
target_names = model_bundle["target_names"]
accuracy = model_bundle["accuracy"]
report = model_bundle["report"]
training_samples = model_bundle["training_samples"]

st.title("🌸 Iris Species Predictor")
st.write(
    "Enter the flower measurements below to predict whether the Iris is "
    "setosa, versicolor, or virginica."
)

st.sidebar.header("Flower Measurements")
sepal_length = st.sidebar.slider(
    "Sepal length (cm)", 4.0, 8.0, 5.8, 0.1
)
sepal_width = st.sidebar.slider(
    "Sepal width (cm)", 2.0, 4.5, 3.0, 0.1
)
petal_length = st.sidebar.slider(
    "Petal length (cm)", 1.0, 7.0, 4.3, 0.1
)
petal_width = st.sidebar.slider(
    "Petal width (cm)", 0.1, 2.5, 1.3, 0.1
)

input_data = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=feature_names,
)

st.subheader("Current measurements")
st.dataframe(input_data, width="stretch", hide_index=True)

if st.button("Predict species", type="primary", width="stretch"):
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    species = target_names[prediction].title()

    st.success(f"Predicted species: **{species}**")

    probability_data = pd.DataFrame(
        {"Species": target_names, "Probability": probabilities}
    ).set_index("Species")
    st.subheader("Prediction confidence")
    st.bar_chart(probability_data)

st.divider()
st.subheader("Model performance")
metric_column, detail_column = st.columns(2)
metric_column.metric("Validation accuracy", f"{accuracy:.1%}")
detail_column.metric("Training samples", training_samples)

with st.expander("View validation report"):
    report_data = pd.DataFrame(report).transpose().round(3)
    st.dataframe(report_data, width="stretch")

st.caption("Dataset: scikit-learn's built-in Iris dataset | Model: Random Forest")
