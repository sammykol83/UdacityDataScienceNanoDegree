#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# import libraries
import pandas as pd
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle

# load data from database
engine = create_engine('sqlite:///DisasterResponse.db')
df = pd.read_sql_table('Message', engine)
X = df['message']
Y = df[df.columns.tolist()[5:]]

# ### 2. Write a tokenization function to process your text data
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [ lemmatizer.lemmatize(token).lower() for token in tokens if token.isalpha()]
    return clean_tokens


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification
# results on the other 36 categories in the dataset. You may find the
# [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html)
# helpful for predicting multiple target variables.
pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline
X_train, X_test, y_train, y_test = train_test_split(X, Y)
#       pipeline.fit(X_train, y_train)

# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating
# through the columns and calling sklearn's `classification_report` on each.
#       y_pred = pipeline.predict(X_test)
#       for i,column in enumerate(y_test.columns.tolist()):
#           print('Column name: %s' % (column))
#           print(classification_report(y_test[column].values, y_pred[:,i]))

# ### 6. Improve your model
# Use grid search to find better parameters. 
parameters = {
    'tfidf__use_idf': (True, False),
    'clf__estimator__n_estimators': [10, 100]
}

cv = GridSearchCV(pipeline, param_grid=parameters)
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)

# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  Since this project focuses on code quality,
# process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure
# to fine tune your models for accuracy, precision and recall to make your project stand out,
# especially for your portfolio!

for i,column in enumerate(y_test.columns.tolist()):
    print('Column name: %s' % (column))
    print(classification_report(y_test[column].values, y_pred[:,i]))

# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF


# ### 9. Export your model as a pickle file
filename = 'classifier.pkl'
pickle.dump(cv, open(filename, 'wb'))

# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a
# database and export a model based on a new dataset specified by the user.
