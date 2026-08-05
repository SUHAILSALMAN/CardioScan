# CardioScan – Heart Disease Risk Prediction System

CardioScan is a Flask-based web application developed as part Computational Intelligence project. The system uses a pre-trained Artificial Neural Network (ANN) to predict the probability of heart disease based on thirteen clinical parameters.

The application provides a healthcare-themed user interface where users can enter patient information and receive an AI-assisted cardiovascular risk assessment, including the predicted outcome, probability of heart disease, and model confidence.

---

# Project Structure

```
HeartDiseaseFlaskApp/
│
├── app.py                     # Flask application
├── heart_disease_model.keras  # Trained ANN model
├── scaler.pkl                 # Saved StandardScaler
├── requirements.txt
├── README.md
│
├── templates/
│     └── index.html
│
└── static/
      ├── style.css
      └── script.js
```

---

# Software Requirements

The project was developed and tested using:

- Python 3.13
- Flask 3.1.1
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Joblib
- HTML5
- CSS3
- JavaScript
- Bootstrap 5

Recommended operating systems:

- Windows 10/11
- Linux
- macOS

Recommended web browsers:

- Google Chrome
- Microsoft Edge
- Mozilla Firefox

---

# Python Libraries

The application requires the following Python libraries:

- Flask
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- Joblib

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

# Installation

### 1. Clone or download the project

Download or clone the project folder to your local machine.

---

### 2. Create a virtual environment (Recommended)

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install required libraries

```bash
pip install -r requirements.txt
```

---

# Running the Application

Start the Flask application by running:

```bash
python app.py
```

After the application starts successfully, open the following URL in your web browser:

```
http://127.0.0.1:5000
```

The trained neural network (`heart_disease_model.keras`) and the saved feature scaler (`scaler.pkl`) are automatically loaded when the application starts.

---

# Required Files

The following files must be present before running the application:

- app.py
- heart_disease_model.keras
- scaler.pkl
- requirements.txt
- templates/index.html
- static/style.css
- static/script.js

Without these files the application cannot run correctly.

---

# Input Features

The model predicts heart disease using the following thirteen clinical features.

| Feature | Description |
|----------|-------------|
| Age | Patient age (years) |
| Sex | Male / Female |
| Chest Pain Type | Type of chest pain |
| Resting Blood Pressure | Blood pressure (mm Hg) |
| Serum Cholesterol | Cholesterol level (mg/dl) |
| Fasting Blood Sugar | Greater than 120 mg/dl (Yes/No) |
| Resting ECG Results | Electrocardiographic findings |
| Maximum Heart Rate | Maximum heart rate achieved |
| Exercise-Induced Angina | Yes / No |
| ST Depression | ST depression induced by exercise |
| Slope of Peak Exercise ST | Upsloping / Flat / Downsloping |
| Number of Major Vessels | 0–3 vessels coloured by fluoroscopy |
| Thallium Stress Test | Normal / Fixed defect / Reversible defect |

These features follow the same order used during model training.

---

# Prediction Workflow

The prediction process consists of the following steps:

1. The user enters patient clinical information.
2. The application validates all input values.
3. The saved StandardScaler preprocesses the input data.
4. The trained Artificial Neural Network predicts the probability of heart disease.
5. The application displays:
   - Predicted outcome (Presence or Absence of Heart Disease)
   - Model confidence
   - Estimated probability of heart disease

The application uses a pre-trained neural network to generate predictions. The model is loaded at startup and used only for prediction.

---

# Main Technologies Used

| Technology | Purpose |
|------------|---------|
| Flask | Web application framework |
| TensorFlow / Keras | Artificial Neural Network inference |
| Scikit-learn | Feature scaling |
| Pandas | Data processing |
| NumPy | Numerical computation |
| Joblib | Loading the trained scaler |
| Bootstrap 5 | User interface design |
| JavaScript | Client-side interaction |

---

## Notes

- CardioScan utilises a pre-trained Artificial Neural Network (ANN) to perform cardiovascular risk prediction based on thirteen clinical parameters.
- Model training and evaluation are conducted separately within the accompanying Jupyter Notebook, while the Flask application is responsible solely for deploying the trained model for inference.
- The application automatically loads the trained neural network (`heart_disease_model.keras`) and the fitted feature scaler (`scaler.pkl`) during system startup to ensure consistent preprocessing and prediction.
- All user inputs undergo comprehensive client-side and server-side validation before being processed by the prediction engine, ensuring data integrity and reliable model inference.
- The application generates AI-assisted cardiovascular risk assessments by reporting the predicted outcome, estimated probability of heart disease, and the corresponding model confidence.
- CardioScan is intended exclusively as an educational and clinical decision-support demonstration. It is not a certified medical device and should not be used as a substitute for professional medical diagnosis, treatment, or clinical judgement.
---

# Suhail Salman

Project Name: **CardioScan – Heart Disease Risk Prediction System**
