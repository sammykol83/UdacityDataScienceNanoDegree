from flask import Flask
from flask import render_template, request, flash
from datetime import datetime
import re
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import fasttext
import random

app = Flask(__name__)
app.secret_key = 'asrtarstaursdlarsn'

class ReusableForm(Form):
    phrase = TextField('Phrase:', validators=[validators.required()])

def Predict(phrase, result_dict):
    return results_dict[random.randint(1,len(results_dict))]

# Dictionary with label options
results_dict = {1:'DBot thins that it''s not a weather or delay event', 
2:'DBot thins that it''s a weather only event', 
3:'DBot thins that it''s a delay only event', 
4:'DBot thins that it''s a delay caused by weather event'}

@app.route("/", methods=['GET', 'POST'])
def home():
     form = ReusableForm(request.form)

     if request.method == 'POST':
        phrase = request.form['phrase'] # Save this sentence to be attached with a tag later on

        # Runs the prediction model
        flash(Predict(phrase, results_dict))

     return render_template(
        "downer_main.html",
        form = form
    )
