from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and preprocessing tools
model = joblib.load('models\\model.pkl')
scaler = joblib.load('models\\scaler.pkl')
venue_encoder = joblib.load('models\\venue_encoder.pkl')
batting_team_encoder = joblib.load('models\\batting_team_encoder.pkl')
bowling_team_encoder = joblib.load('models\\bowling_team_encoder.pkl')
striker_encoder = joblib.load('models\\striker_encoder.pkl')
bowler_encoder = joblib.load('models\\bowler_encoder.pkl')

# Load options (for dropdowns)
df = joblib.load('models\\dropdown_data.pkl')

@app.route('/')
def index():
    return render_template('index.html',
                           venues = sorted(set(df['venues'])),
                           batting_teams=sorted(set(df['batting_teams'])),
                           bowling_teams=sorted(set(df['bowling_teams'])),
                           batsmen=sorted(set(df['strikers'])),
                           bowlers=sorted(set(df['bowlers'])))

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        venue = request.form['venue']
        bat_team = request.form['batting_team']
        bowl_team = request.form['bowling_team']
        batsman = request.form['striker']
        bowler = request.form['bowler']

        # Encode inputs
        input_array = np.array([
            venue_encoder.transform([venue])[0],
            batting_team_encoder.transform([bat_team])[0],
            bowling_team_encoder.transform([bowl_team])[0],
            striker_encoder.transform([batsman])[0],
            bowler_encoder.transform([bowler])[0]
        ]).reshape(1, -1)

        # Scale
        input_scaled = scaler.transform(input_array)

        # Predict
        prediction = int(model.predict(input_scaled)[0][0])

        return render_template('index.html',
                        venues = sorted(set(df['venues'])),
                        batting_teams=sorted(set(df['batting_teams'])),
                        bowling_teams=sorted(set(df['bowling_teams'])),
                        batsmen=sorted(set(df['strikers'])),
                        bowlers=sorted(set(df['bowlers'])),
                        predicted_score=prediction)

if __name__ == '__main__':
    app.run(debug=True)
