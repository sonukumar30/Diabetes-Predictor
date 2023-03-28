import io
import base64
from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
app = Flask(__name__,static_folder="/home/sonu/Downloads/Diabetes-Predictor-main/templates/static")
model = pickle.load(open("model_pickle", "rb"))



@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")

@app.route('/about-us')
def about_us():
    return render_template('index.html')


@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        preg = request.form["Preg"]
        Glucose = request.form["Glucose"]
        BloodPressure = request.form["BloodPressure"]
        SkinThickness = request.form["SkinThickness"]
        Insulin = request.form["Insulin"]
        BMI = request.form["BMI"]
        Dia = request.form["Dia"]
        Age = request.form["Age"]

        #print(Age, Dia, BMI, Insulin, SkinThickness, BloodPressure, Glucose, preg)
        prediction = model.predict([[preg, Glucose, BloodPressure, SkinThickness, Insulin, BMI, Dia, Age]])
        
        if prediction[0] == 1:
            output = "High"
            color = 'red'
        else:
            output = "Low"
            color = 'green'

        # Create a bar graph to display the prediction
        fig, ax = plt.subplots(figsize=(6, 6))
        x_values = ['No Diabetes', 'Diabetes']
        y_values = [1 - prediction[0], prediction[0]]
        ax.bar(x_values, y_values, color=color)

        # Customize the graph
        plt.ylim(0, 1)
        plt.title('Chances of having Diabetes')
        plt.xlabel('Diabetes Status')
        plt.ylabel('Probability')

        # Convert the chart to a base64 encoded string
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        chart = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return render_template('home.html', prediction_text="Chances of having Diabetes is {}".format(output), chart=chart)


if __name__ == "__main__":
    app.run(debug=True)