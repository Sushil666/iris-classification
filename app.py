import streamlit as st
import pandas as pd
import joblib

# Load dataset (for slider ranges only)
df = pd.read_csv("iris.csv")

# Cache model loading (important for performance)
@st.cache_resource
def load_model():
    return joblib.load("knn_model.joblib")

model = load_model()

st.title("🌸 Iris Flower Species Prediction App")
st.write("This app predicts the species of an Iris flower based on its features.")

st.sidebar.header("Input Flower Measurements")

sepal_length = st.sidebar.slider(
    "Sepal Length (cm)",
    float(df['SepalLengthCm'].min()),
    float(df['SepalLengthCm'].max())
)

sepal_width = st.sidebar.slider(
    "Sepal Width (cm)",
    float(df['SepalWidthCm'].min()),
    float(df['SepalWidthCm'].max())
)

petal_length = st.sidebar.slider(
    "Petal Length (cm)",
    float(df['PetalLengthCm'].min()),
    float(df['PetalLengthCm'].max())
)

petal_width = st.sidebar.slider(
    "Petal Width (cm)",
    float(df['PetalWidthCm'].min()),
    float(df['PetalWidthCm'].max())
)

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

prediction = model.predict(input_data)

st.sidebar.subheader("🌼 Prediction Result")
st.sidebar.success(f"Predicted Species: {prediction[0]}")
