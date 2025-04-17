# IPL Score Predictor

This project leverages deep learning techniques to predict scores in Indian Premier League (IPL) cricket matches. The codebase includes data preprocessing, training of a neural network, and an interactive widget for score prediction based on user inputs.

## Features

- **Data Preprocessing**: The dataset contains information from IPL matches (2008-2017) including teams, players, and match statistics. Features are preprocessed using techniques like feature scaling, label encoding, and train-test split.
- **Deep Learning Model**: A neural network is implemented using TensorFlow and Keras for regression tasks, predicting the total runs scored.
- **Interactive Widget**: A user-friendly interface to input match conditions (venue, batting team, bowling team, striker, and bowler) and get score predictions.

## Steps in the Notebook

1. **Import Libraries**: Use libraries like `pandas`, `numpy`, `scikit-learn`, `tensorflow`, and `ipywidgets`.
2. **Data Loading and Preprocessing**:
   - Features like venue, teams, and players are encoded.
   - Data is scaled and split into training and testing sets.
3. **Model Definition**: A feedforward neural network with hidden layers is defined and compiled using the Huber loss function.
4. **Model Training**: The model is trained with the training dataset to predict the total scores.
5. **Model Evaluation**: The model's performance is evaluated using metrics like Mean Absolute Error (MAE).
6. **Interactive Score Prediction**:
   - An interactive widget allows users to select match conditions.
   - The trained model predicts the total runs based on the inputs.

## Requirements

- Python 3.7 or above
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `tensorflow`
  - `ipywidgets`

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/saksham3232/IPL-Score-Predictor.git
   cd IPL-Score-Predictor
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook "IPL Score Predictor.ipynb"
   ```
4. Follow the steps in the notebook to preprocess the data, train the model, and use the interactive widget for score prediction.

## Dataset

The dataset includes features such as:
- Venue
- Batting and bowling teams
- Player names
- Runs, wickets, and overs

## Results

- The trained model achieves competitive accuracy in predicting the total runs scored in IPL matches.
- The interactive widget provides a dynamic way to predict scores for specific match conditions.

## Acknowledgements

- This project uses IPL data from 2008 to 2017.
- Built with TensorFlow and Keras for deep learning.
