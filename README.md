# Iris Classification

A machine learning application for classifying Iris flower species based on their features using K-Nearest Neighbors (KNN) algorithm.

## Project Overview

This project uses the Iris dataset to train a KNN classifier that predicts the species of an Iris flower based on four features:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

The trained model classifies flowers into three species:
- Setosa
- Versicolor
- Virginica

## Project Structure

```
iris-classification/
├── app.py                 # Streamlit web application for predictions
├── iris.csv              # Iris dataset
├── knn_model.joblib      # Trained KNN model
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Features

- **Interactive Web Interface**: Built with Streamlit for easy-to-use predictions
- **Real-time Predictions**: Input flower measurements and get instant species predictions
- **Pre-trained Model**: Uses a KNN classifier with k=6 for accurate predictions
- **User-friendly Sliders**: Select flower measurements using intuitive slider inputs

## Requirements

- Python 3.7+
- See `requirements.txt` for all dependencies

## Installation

1. Clone or download this project
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

Then open your web browser to the local URL (typically `http://localhost:8501`)

Use the sidebar sliders to input the iris flower measurements and receive the species prediction.

## Model Information

- **Algorithm**: K-Nearest Neighbors (KNN)
- **Number of Neighbors (k)**: 6
- **Training/Test Split**: 80/20
- **Random State**: 123

## Dataset

The iris.csv file contains measurements of iris flowers with the following columns:
- SepalLengthCm
- SepalWidthCm
- PetalLengthCm
- PetalWidthCm
- Species (target variable)

## License

This project is open source and available for educational purposes.
