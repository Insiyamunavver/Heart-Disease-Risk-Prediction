# Heart-Disease-Risk-Prediction
Streamlit-based clinical decision support system for heart disease risk prediction using machine learning.
## Overview
This application allows users to input patient health information and receive a probability-based assessment of heart disease risk.  


## Features
- Risk prediction using a trained machine learning model
- Probability-based output with clear risk interpretation
- Visualization of key contributing risk factors
- Clean, clinical-style user interface
- Deployed as a web application using Streamlit

## Model & Approach
- Multiple classification models were trained and evaluated offline.
- The best-performing model was selected and saved for inference.
- The deployed application uses the pre-trained model and scaler for real-time predictions.
- Model training is **not performed** inside the web application.

## Tech Stack
- Python
- Streamlit
- Scikit-learn
- NumPy
- Pandas

