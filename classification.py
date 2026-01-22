import joblib
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,ConfusionMatrixDisplay


df = pd.read_csv("iris.csv")

X = df.drop(columns=['Species'])
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
y_preds = knn.predict(X_test)

joblib.dump(knn, 'knn_model.joblib')

st.title("Iris Flower Species Prediction App")
st.write("This app predicts the species of an Iris flower based on its features.")

st.sidebar.title("Iris Flower Species Prediction")

sepal_length = st.sidebar.slider("Sepal Length", float(df['SepalLengthCm'].min()), float(df['SepalLengthCm'].max()))
sepal_width = st.sidebar.slider("Sepal Width", float(df['SepalWidthCm'].min()), float(df['SepalWidthCm'].max()))
petal_length = st.sidebar.slider("Petal Length", float(df['PetalLengthCm'].min()), float(df['PetalLengthCm'].max()))
petal_width = st.sidebar.slider("Petal Width", float(df['PetalWidthCm'].min()), float(df['PetalWidthCm'].max()))        

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

model = joblib.load('knn_model.joblib')
prediction = model.predict(input_data)

st.sidebar.subheader("Prediction Result")
st.sidebar.write(f"The predicted species is: {prediction[0]}")  