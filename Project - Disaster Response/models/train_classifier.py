import sys
import pickle
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

def load_data(database_filepath):
    """loads messages and categories from the database and inserts them into pandas frames.
       Args:
       database_filepath (str): path for loading the SQL databse
       Returns:
       X (DataFrame): Messages.
       Y (DataFrame): Categories.
       category_names(List): List of category names strings
    """
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('InsertTableName', engine)
    X = df['message']
    category_names = df.columns.tolist()[5:]
    Y = df[category_names]
    return X, Y, category_names


def tokenize(text):
    """ Applies word tokenization --> WordNet lemmatization over messages transformed to lowercase and
        with only alphanumeric characters.
        Args:
        text (str): String to tokenize
        Returns:
        clean_tokens (list): List of clean words
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(token).lower() for token in tokens if token.isalpha()]
    return clean_tokens


def build_model():
    """ Builds a pipeline machine learning model (for NLP application) with the following steps:
        Lemmatized text (input) --> CountVectorizer --> TfidTransformer --> MultiOutputClassifier
        where for each label, the classifier is a random forest, which is by itself an ensemble
        of decision trees. Model parameters are optimized via grid search.

        Args: None.
        Returns: optimized model.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [10, 100]
    }

    cv = GridSearchCV(pipeline, param_grid = parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """ Passes a test set through the ML pipeline and compares the predicted results with the known results.
        Uses the classification_report function from sklearn.metrics to get a report of performance.

        Args:
        model: The ML pipeline generated model.
        X_test (DataFrame): Frame containing the test set messages.
        Y_test (DataFrame): Frame containing the labels (categories) for the X_test data.
        category_names (list): A list of the category names.
        Returns: None.
    """
    y_pred = model.predict(X_test)
    for i, column in enumerate(category_names):
        print('Column name: %s' % column)
        print(classification_report(Y_test[column].values, y_pred[:, i]))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()