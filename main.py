import os
from flask import Flask, render_template, request
import pickle
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))

# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True


def load_data():
    matches_till_2022 = pd.read_csv('Dataset/Processed_Data/Matches_Till_2022.csv')
    return matches_till_2022


match = load_data()


def load_models():
    LogReg = pickle.load(open('LogReg.pkl', 'rb'))
    Rf = pickle.load(open('Rf.pkl', 'rb'))
    dt_clf = pickle.load(open('dt_clf.pkl', 'rb'))
    return LogReg, Rf, dt_clf


LogReg, Rf, dt_clf = load_models()


def getPrediction(batting_team, bowling_team, selected_city, target, score, overs, wickets_out, classifier_name):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets_out
    crr = score / overs
    rrr = (runs_left * 6) / balls_left

    input_df = pd.DataFrame({'Batting_Team': [batting_team], 'Bowling_team': [bowling_team],
                             'City': [selected_city], 'runs_left': [runs_left], 'balls_left': [balls_left],
                             'wickets': [wickets_left], 'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]})

    if classifier_name == "Logistic Regression":
        result = LogReg.predict_proba(input_df)
    elif classifier_name == "Random Forest":
        result = Rf.predict_proba(input_df)
    else:
        result = dt_clf.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    return win, loss

@app.route("/")
def startPage():
    return render_template("1stpage.html")


@app.route('/predict', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        iplFormData = request.get_json()
        print(iplFormData)
        batting_team = iplFormData['batting_team']
        bowling_team = iplFormData['bowling_team']
        selected_city = iplFormData['selected_city']
        target = iplFormData['target']
        score = iplFormData['score']
        overs = iplFormData['overs']
        wickets_out = iplFormData['wickets_out']
        classifier_name = iplFormData['classifier_name']
        winingProbability, lossingProbability = getPrediction(batting_team, bowling_team, selected_city, target, score, overs, wickets_out, classifier_name)
        print("winingProbability =", winingProbability)
        print("lossingProbability =", lossingProbability)
        return winingProbability, lossingProbability
    return render_template('riderLogin.html')