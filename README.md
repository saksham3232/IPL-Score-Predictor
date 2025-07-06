# IPL Score Predictor

Predict the final score of an IPL (Indian Premier League) cricket match using a Deep Learning model trained on historical IPL data!

This project leverages a neural network (TensorFlow/Keras) to estimate the total score based on current match context—venue, teams, batsman, and bowler. It features an interactive web app (Flask) and a Jupyter Notebook for data science exploration.

---

## Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Setup Instructions](#setup-instructions)
  - [Run the Web App](#run-the-web-app)
- [How it Works](#how-it-works)
- [Model Evaluation](#model-evaluation)
- [References](#references)
- [License](#license)

---

## Features

- Predict final IPL scores based on chosen venue, teams, batsman, and bowler.
- Trained on IPL data from 2008 to 2017.
- End-to-end notebook with data cleaning, feature engineering, model training, and evaluation.
- Interactive web app using Flask for easy predictions.
- Categorical encoders and scaler saved for robust inference.

---

## Demo

![Web App Screenshot](assets/screenshot.png) <!-- Replace with your screenshot path -->

---

## Project Structure

```
.
├── IPL Score Predictor.ipynb   # Jupyter notebook: data prep, model training, EDA
├── app.py                     # Flask web app for predictions
├── ipl_data.csv               # IPL delivery-level historical data
├── models/
│   ├── model.pkl              # Saved Keras model
│   ├── scaler.pkl             # MinMaxScaler for feature scaling
│   ├── venue_encoder.pkl      # Label encoders for categorical inputs
│   ├── batting_team_encoder.pkl
│   ├── bowling_team_encoder.pkl
│   ├── striker_encoder.pkl
│   ├── bowler_encoder.pkl
│   └── dropdown_data.pkl      # Dropdown options for web app
└── templates/
    └── index.html             # Web app UI (not shown here)
```

---

## Getting Started

### Requirements

- Python 3.7+
- pip (Python package manager)

**Main libraries:**
- pandas
- numpy
- scikit-learn
- tensorflow / keras
- flask
- joblib
- ipywidgets (for notebook only)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/saksham3232/IPL-Score-Predictor.git
   cd IPL-Score-Predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   If `requirements.txt` is missing, install manually:
   ```bash
   pip install pandas numpy scikit-learn tensorflow flask joblib ipywidgets
   ```

3. **(Optional) Train the Model**
   - Open `IPL Score Predictor.ipynb` in Jupyter Notebook.
   - Run all cells to preprocess, train, and save the model.
   - The notebook will create all necessary files in the `models/` directory.

---

### Run the Web App

1. **Ensure all model files are present in the `models/` directory.**
2. **Start the Flask app**
   ```bash
   python app.py
   ```
3. **Visit [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your web browser.**
4. **Select the match context and get your score prediction!**

---

## How it Works

1. **Data Processing**
   - Loads IPL ball-by-ball data (`ipl_data.csv`).
   - Drops irrelevant columns and encodes categorical variables using LabelEncoder.
   - Splits data into train/test sets and applies MinMax scaling.

2. **Model**
   - A deep neural network (Keras Sequential) with two hidden layers:
     - 512 and 216 neurons, both with ReLU activation.
   - Output layer predicts the total score (regression).
   - Compiled with Adam optimizer and Huber loss (robust to outliers).

3. **Evaluation**
   - Metrics: MAE, MSE, RMSE, R².
   - Training/validation loss is plotted.

4. **Web App**
   - User selects venue, batting team, bowling team, batsman, and bowler.
   - Inputs are encoded and scaled using saved transformers.
   - The trained model predicts the final score.

---

## Model Evaluation

Example metrics (may vary with retraining):

- **Mean Absolute Error (MAE):** ~19.2
- **Root Mean Squared Error (RMSE):** ~26.3
- **R² Score:** ~0.18

---

## References

- Data: [Kaggle IPL Ball-by-Ball Dataset](https://www.kaggle.com/datasets/nowke9/ipl-data-set)
- Keras Documentation: https://keras.io/
- scikit-learn Documentation: https://scikit-learn.org/

---

**Author:** [Saksham](https://github.com/saksham3232)
