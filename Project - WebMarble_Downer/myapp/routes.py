import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from sklearn.externals import joblib
from sqlalchemy import create_engine
from return_figures import return_figures

#----------- Change this flag to run on a local PC host --------- #
run_on_udacity_terminal = 0
#---------------------------------------------------------------- #

app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('InsertTableName', engine)

# load model
model = joblib.load("../models/classifier.pkl")

figures = return_figures()

# plot ids for the html id tag
ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index2')
def index2():

    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('master.html', ids=ids, figuresJSON=figuresJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results,
        ids=ids
    )


def main():
    # Decide IP to use according to platform
    if run_on_udacity_terminal:
        host_ip = '0.0.0.0'
    else:
        host_ip = '127.0.0.1'
    app.run(host=host_ip, port=3001, debug=True)


if __name__ == '__main__':
    main()