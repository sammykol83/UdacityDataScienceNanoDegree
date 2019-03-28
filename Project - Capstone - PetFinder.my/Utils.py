# Imports
import pandas as pd
import numpy as np
import json
import os
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import string 
import glob
from sklearn.model_selection import learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from xgboost import XGBClassifier
import xgboost as xgb
import lightgbm as lgb

# Cohen Kappa score metric exists in sklearn. See discussion here:
# https://www.kaggle.com/aroraaman/quadratic-kappa-metric-explained-in-5-simple-steps
from sklearn.metrics import cohen_kappa_score

# Download word2vec corpus + related nlp modules
import nltk, gensim
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('word2vec_sample')
nltk.download('stopwords')

# Imports related to neural network processing of images
import torch
from torch import nn
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch import optim
from PIL import Image
from torch.autograd import Variable

#============================================================================================= #
#============================ Functions related to ETL pipeline/NLP ========================== #

def GetDescriptionSentiment(filename):
    """
    Gets the json filename and returns the sentiment as a weighted average of all sentiments.
    (This means sum of scores times magnitude divided by the number of sentences in the description)

    Parameters
    ----------
    filename : string with format '{PetID}.json', e.g: '000fb9572.json'

    Returns
    ----------
    w_score: Weighted score for all sentences in the description
    """
    
    # Open and read file
    fullpath = os.getcwd() + '\\Data\\train_sentiment\\' + filename
    try:
        with open(fullpath, errors='ignore') as f:
            description = json.load(f)
    except:
        return 0
    
    # Collect all scores and magnitudes
    scores, magnitudes = [],[]
    for sentence in description['sentences']:
        scores.append(sentence['sentiment']['score'])
        magnitudes.append(sentence['sentiment']['magnitude'])
        
    # Weighted score is the weighted average of scores
    w_score = 0
    L = len(scores)
    for i in range(0,L):
        w_score += (scores[i] * magnitudes[i])/L
    
    return w_score

def DoTFIDforDescriptions(df, n_components):
    """
    Takes all descriptions from input dataframe, converts to numeric vector according the the TF-IDF model
    (see: https://en.wikipedia.org/wiki/Tf%E2%80%93idf) and then reduces the dimensionality via SVD.
    I quote:"In particular, truncated SVD works on term count/tf-idf matrices as returned by the vectorizers in sklearn.feature_extraction.text. 
    In that context, it is known as latent semantic analysis (LSA)" from (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)
    
    Parameters
    ----------
    df : (Pandas dataframe) A dataframe containing textual descriptions.
    n_components: (int) - The number of required dimensions to reduce the TF-IDF matrix into. 
    
    Returns
    ----------
    df with concatenated new components extracted from the text description. 
    """
    # Fill nothing for missing descriptions
    desc = df.Description.fillna("").values
    
    ## Define TF-IDF vectorizer
    # min_df=3: ignore terms appearing less than 3 times. 
    # max_features=10000: build a vocabulary that only consider the top 10000 ordered by term frequency across the corpus.
    # strip_accents='unicode': Strip accents for "weird" charachters
    # analyzer=word: Create corpus from words
    # token_pattern: Regular expression denoting what constitutes a “token".
    # ngram_range: best exaplained here: https://www.kaggle.com/c/avito-demand-prediction/discussion/58819
    # stop_words: English, though some messages are NOT in english.. 
    tfv = TfidfVectorizer(min_df=3,  max_features=10000,
            strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
            ngram_range=(1, 3), stop_words = 'english')

    # Fit TFIDF (Train)
    tfv.fit(list(desc))
    
    # Convert to numeric vector
    X = tfv.transform(desc)
    
    # Vector is too long, we truncate to primary components
    svd = TruncatedSVD(n_components=n_components)
    Xr = svd.fit_transform(X)
    
    # Generate a dataframe from these components
    X = pd.DataFrame(Xr, columns=['svd_{}'.format(i) for i in range(n_components)])
    
    # Attach to our input DF 
    df = pd.concat((df, X), axis=1)
    
    return df

def EmbedDescription(description, model):
    """
    Converts a text description into a vector via averaging the embedded vectors
    of all words in the description. 
    In the process, we do a nice NLP "pipeline" like we learned in the course:
    Normalization-->tokenization-->removal of stop words-->lemmatization.
    
    Parameters
    ----------
    description : (string) description of adoption
    model: (gensim.models.word2vec.Word2Vec) - A Word2Vec model which basically acts as a dictionary. 
    Input a word, get back a 300 elements vector representing the word. 
    
    Returns
    ----------
    df with concatenated new components extracted from the text description. 
    """
    
    # Initialize vector representation
    vec = np.zeros((300,))
        
    # Convert to lowercase (normalization)
    if (pd.isna(description) == 0):
        description = description.lower()

        # Remove punctuation
        description = description.translate(description.maketrans('', '', string.punctuation))

        # Split to words using tokenizer
        words = nltk.tokenize.word_tokenize( description )

        # Remove English stop words (assuming the text is in English)
        words = [w for w in words if w not in nltk.corpus.stopwords.words("english")]

        # Lemmatize form of words
        lemmed = set([WordNetLemmatizer().lemmatize(w) for w in words])

        # Leave only words of lemmed that are in our vocabulary
        in_vocab_lemmed = []
        for lem in lemmed:
            if lem in model.vocab:
                in_vocab_lemmed.append(lem)    

        # Get vector representation of each word, average all words
        l = len(in_vocab_lemmed)
        for word in in_vocab_lemmed:
            vec += model[word]/l
        
        vec = np.reshape(vec, (300,1))
        
    return vec

def DoEmbeddforDescriptions(df, n_dim = 10):
    """
    Wrapper for EmbedDescription(). Gets a Dataframe, loads and Word2Vec model, goes over all descriptions
    in the dataframe and sends them to the EmbedDescription function which returns their vector embedding.
    Additionally, this embedding is approximated by a lower dimensional vector via SVD. 
    
    Parameters
    ----------
    description : (string) description of adoption
    model: (gensim.models.word2vec.Word2Vec) - A Word2Vec model which basically acts as a dictionary. 
    Input a word, get back a 300 elements vector representing the word. 
    
    Returns
    ----------
    df with concatenated new components extracted from the text description. 
    """
    
    # Init
    N, q = df.shape[0], 0
    petIds = df['PetID'].values.tolist()[0:N]
    X = np.zeros((N,300))
    
    # Load the word2vec corpus
    word2vec_sample = str(nltk.data.find('models/word2vec_sample/pruned.word2vec.txt'))
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)
    
    # For each petID, get a description vector
    for petID in petIds:

        # Print message
        if np.mod(q,100) == 0:
            print('Processed word embedding for %d descriptions of %d...' % (q, len(petIds)))

        # Get description
        description = df[df['PetID'] == petID]['Description'].values[0]

        # Get vector representation
        vec = EmbedDescription(description, model)

        # Add these features to our data
        X[q,:] = vec.T
        q += 1
    
    # Reduce dimensionality via SVD
    svd = TruncatedSVD(n_components=n_dim)
    Xr = svd.fit_transform(X)
    
    # Generate a DataFrame from reduced representation of description
    str_list = []
    for idx in range(0, 10):
        str_list.append('v_' + str(idx))
    desc_embedding_df = pd.DataFrame(data = Xr, columns = str_list)
    desc_embedding_df.set_index(df.index.values[0:N], inplace=True, verify_integrity=False)
    
    # Concat to input df
    df = pd.concat((df, desc_embedding_df), axis=1)
    
    return df

def ETL_pipeline(filename, is_test_set, lgb_clf_flag, do_additional_nlp):
    """
    Converts the data from file to pandas dataframe with original and additional columns for using in the 
    ML algorithms for classifcation.
    
    Parameters
    ----------
    filename : (string) filename of data. (Should be the "train.csv" or "test.csv")
    is_test_set: (bool) - If True, labels won't be returned. (Because we don't have them...)
    lgb_clf_flag: (bool) - If True, it means we're planning to work with the LightGBM classifier and it needs 
    to know which columns are the categorical ones. 
    NOTICE: A better software practice is to do this modification at some other point in the code (in my opinion..)
	do_additional_nlp: (bool) if True, will add a vector representation for the "description" field of the data. 
    
    Returns
    ----------
    df_dogs: (Pandas DF) Dataframe with all data only for dogs
    df_cats: (Pandas DF) Dataframe with all data only for cats
    labels_dogs: (Pandas DF) Adoption speeds for dogs
    labels_cats:(Pandas DF) Adoption speeds for cats
    """
    
    # Init
    columns_to_drop = ['Name','RescuerID']
    nlp_method = 'tf-idf' # options are:'tf-idf' or 'embedd' 
    DOG, CAT = 1,2
 
    ## Load the data
    df = pd.read_csv(os.getcwd() + '\\Data\\' + filename)

    ## Drop non relevant columns (in my opinion)
    df.drop(columns_to_drop, axis=1, inplace=True)

    ## Drop rows with missing values
    # ASSUMPTION: Negligble number of such rows.
    # Todo: According to train/test data, missing values only in description. Can be handled differently since description
    # is later replaced by sentiment or other embedding...
    #indices_with_nan = df[pd.isnull(df).any(axis=1)].index.values
    #df.drop(labels=indices_with_nan, inplace=True)

    ## No normalization / scaling since classifier isn't going to be distance based (no Neural Network)
    
    ## No duplicates handling (I checked on train/test data)

    ## Replace description with sentiment
    sentiments = []
    for rowIdx in range(0,df.shape[0]):
        sentiments.append(GetDescriptionSentiment(df.iloc[rowIdx]['PetID']+'.json'))

    # Concatenate the sentiments to our dataframe, then drop the
    sentiments_df = pd.DataFrame({'sentiment':sentiments}, index=df.index.values)
    df = pd.concat([df, sentiments_df], axis=1, join='outer')
    
    # Add columns from NLP
    if (do_additional_nlp == True):
        if (nlp_method == 'tf-idf'):
            df = DoTFIDforDescriptions(df, 10)
        elif (nlp_method == 'embedd'):
            df = DoEmbeddforDescriptions(df, 10)
    df.drop(labels=['Description'], inplace=True, axis=1)
    
    # Change columns to category for lgb classifier, if required
    if lgb_clf_flag == True:
        cat_cols = ['Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health','State']
        for col in cat_cols:
            df[col] = df[col].astype('category')
    
    # Split to dogs/cats data frames. (I leave the 'Type' column for later...)
    df_dogs = df[df['Type'] == DOG]
    df_cats = df[df['Type'] == CAT]
    
    ## Generate dataframe for labels
    if (is_test_set == 0):
        labels_dogs = pd.DataFrame(df_dogs['AdoptionSpeed'], columns=['AdoptionSpeed']) # This is a dataframe
        df_dogs.drop('AdoptionSpeed', axis=1, inplace=True)
        labels_cats = pd.DataFrame(df_cats['AdoptionSpeed'], columns=['AdoptionSpeed']) # This is a dataframe
        df_cats.drop('AdoptionSpeed', axis=1, inplace=True)
    else:
        labels_dogs, labels_cats = [],[]

    return df_dogs, df_cats, labels_dogs, labels_cats

def ETL_pipeline_ver2(filename, is_test_set, lgb_clf_flag, do_additional_nlp):
    """
    2nd version of the function. 
    Converts the data from file to pandas dataframe with original and additional columns for using in the 
    ML algorithms for classifcation. 
    It works with the TF-IDF method for NLP, uses sentiment analysis and generates a single dataframe for BOTH dogs and cats.
	
    Parameters
    ----------
    filename : (string) filename of data. (Should be the "train.csv" or "test.csv")
    is_test_set: (bool) - If True, labels won't be returned. (Because we don't have them...)
    lgb_clf_flag: (bool) - If True, it means we're planning to work with the LightGBM classifier and it needs 
    to know which columns are the categorical ones. 
    NOTICE: A better software practice is to do this modification at some other point in the code (in my opinion..)
	do_additional_nlp: (bool) if True, will add a vector representation for the "description" field of the data.
    
    Returns
    ----------
    df: (Pandas DF) Dataframe with all data for both dogs AND cats (contrary to 1st version)
    labels: (Pandas DF) Adoption speeds for both dogs AND cats (contrary to 1st version) 
    """
    # Init
    columns_to_drop = ['Name','RescuerID']
    nlp_method = 'tf-idf' # options are:'tf-idf' or 'embedd' 

    ## Load the data
    df = pd.read_csv(os.getcwd() + '\\Data\\' + filename)

    ## Drop non relevant columns (in my opinion)
    df.drop(columns_to_drop, axis=1, inplace=True)

    ## Replace description with sentiment
    sentiments = []
    for rowIdx in range(0,df.shape[0]):
        sentiments.append(GetDescriptionSentiment(df.iloc[rowIdx]['PetID']+'.json'))

    # Concatenate the sentiments to our dataframe, then drop the
    sentiments_df = pd.DataFrame({'sentiment':sentiments}, index=df.index.values)
    df = pd.concat([df, sentiments_df], axis=1, join='outer')

    # Add columns from NLP
    if (do_additional_nlp == True):
        if (nlp_method == 'tf-idf'):
            df = DoTFIDforDescriptions(df, 10)
        elif (nlp_method == 'embedd'):
            df = DoEmbeddforDescriptions(df, 10)
    df.drop(labels=['Description'], inplace=True, axis=1)

    # Change columns to category for lgb classifier, if required
    if lgb_clf_flag == True:
        cat_cols = ['Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health','State']
        for col in cat_cols:
            df[col] = df[col].astype('category')

    ## Generate dataframe for labels
    if (is_test_set == 0):
        labels = pd.DataFrame(df['AdoptionSpeed'], columns=['AdoptionSpeed']) # This is a dataframe
        df.drop('AdoptionSpeed', axis=1, inplace=True)
    else:
        labels = []

    return df, labels

#============================ Functions related to Training/Testing our ML models ========================== #

def PlotEstimatorPerformance(clf, n_folds):
    """
    Plots the training and cross validation scores for a gridSearchCV trained classifier.
    
    Parameters
    ----------
    clf : (gridSearchCV) The classifier AFTER it has been fitted.
    n_folds: (int) - Number of folds to plot. (Must be <= actual number of folds in the cross validation)opinion..)
    
    Returns
    ----------
    none
    """
        
    train_scores, test_scores = [], []
    for foldIdx in range(0,n_folds):
        train_str = 'split'+str(foldIdx)+'_train_score'
        test_str = 'split'+str(foldIdx)+'_test_score'
        if len(clf.cv_results_['rank_test_score']) == 1:
            train_scores.append(clf.cv_results_[train_str][clf.cv_results_['rank_test_score'][0]-1])
            test_scores.append(clf.cv_results_[test_str][clf.cv_results_['rank_test_score'][0]-1])
        else:
            train_scores.append(clf.cv_results_[train_str][clf.cv_results_['rank_test_score'][0]])
            test_scores.append(clf.cv_results_[test_str][clf.cv_results_['rank_test_score'][0]])

    fig, ax = plt.subplots(figsize=(10,5))
    x = list(range(0,n_folds))
    plt.plot(x,train_scores, 'b-o', linewidth = 2, markerSize = 14)
    plt.plot(x,test_scores, 'r-o', linewidth = 2, markerSize = 14)
    plt.grid(True)
    plt.xlabel('Fold index')
    plt.ylabel('QWK score')
    plt.legend(['Training set','CV set'])
    plt.title('Mean Train QWK: %2.5f, Mean CV QWK: %2.5f' % (np.mean(train_scores), np.mean(test_scores)))

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), 
                        scorer = make_scorer(cohen_kappa_score, weights='quadratic')):
    """
    Generate a simple plot of the test and training learning curve.
	The function was copied from the scikit tutorials:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
        
    scorer: A modification I added. Allows a custom scoring metric. (In our case, it will be the QWK)
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring = scorer)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt 

def FindBestClfGridSearch(parameters, X, y, scorer, cv=5, clf_type = 'lgb', plot_learning_curve = True):
    """
    Receives data and labels. Splits to training and testing sets, initializes a grid search of the parameters
    of a selected classifier, plots the training results and returns the classifier and the testing data.
    
    Parameters
    ----------
    parameters : (dict) dictionary of hyperparameters names and corresponding list of values for testing in the
                 gridsearchCV function. 
    X: (array like) - Training data.
    y: (array like) - Labels data. 
    scorer: (function) - A function that implements a scoring metric for the classifier to use.
    cv: (int) - Number of required K folds in cross validation for classifier.
    clf_type (string) - Can either be 'xgb' or 'lgb'. (I prefer lgb.. Faster)
    plot_learning_curve (bool) - If 'True', will calculate and plot the learning curve for the model. 
    
    Returns
    ----------
    clf: (model) - The classification model result of the gridsearchCV.
    X_test: (array like) - The test set data that was created from the data by the train_test_split(...) function. 
    y_test: (array like) - The test labels that were created from the data by the train_test_split(...) function. 
    """
    
    # Split data to train/test sets
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
    
    # Initialize classifier
    if clf_type == 'xgb':
        model = XGBClassifier()
    elif clf_type == 'lgb':
        model = lgb.LGBMClassifier()

    # Initialize grid search with 5 folds
    clf = GridSearchCV(model, param_grid=parameters, scoring = scorer, cv=cv, 
                       verbose=True, return_train_score=True)

    # Train
    clf.fit(X_train, y_train)
    
    # Check performance of dogs classifier
    TestClassifier(clf, X, y, X_test, y_test, cv, plot_learning_curve)

    return clf, X_test, y_test

def TestClassifier(clf, X, y, X_test, y_test, cv=5, plt=True):
    """
    1. Prints the best parameters found using the gridsearchCV. 
    2. Receives a test vector, performs predictions upon it using the best estimator and calculates the QWK score.
    3. If required, also plots the learning curve, so we can understand if our model learns well. 
    
    Parameters
    ----------
    clf: (model) - The classification model result of the gridsearchCV.
    X: (array like) - Training data.
    y: (array like) - Labels data. 
    X_test: (array like) - The test set data that was created from the data by the train_test_split(...) function. 
    y_test: (array like) - The test labels that were created from the data by the train_test_split(...) function. 
    cv: (int) - Number of required K folds in cross validation for classifier.
    plt (bool) - If 'True', will calculate and plot the learning curve for the model. 
    
    Returns
    ----------
    none
    """
    print("Best parameters set found on development set:")
    print(clf.best_params_)
    PlotEstimatorPerformance(clf, cv)
    y_pred = clf.predict(X_test)    
    print('#### Test set cohen_kappa: %2.5f ####' % (cohen_kappa_score(y_test, y_pred, weights='quadratic')))
    if plt == True:
        plot_learning_curve(clf.best_estimator_, 'Classifier Learning curve', X, y, ylim=None, cv=5,
                            n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))

def PlotImportances(df, clf, type):
    """
    Plots the importances of features of a trained classifier
    
    Parameters
    ----------
    clf: (model) - The classification model result of the gridsearchCV.
    df: (DataFrame) - DataFrame. We need it to extract column names from it.
    type: (string) - The required importance type. For example: 'gain' for a tree model. 
    
    Returns
    ----------
    none
    """

    # Get the feature importances (sorted, from high to low)
    clf.best_estimator_.importance_type = type
    sorted_importances = np.sort(clf.best_estimator_.feature_importances_)

    # Get the feature importances INDICES (sorted, from high to low)
    sorted_idxs = np.argsort(clf.best_estimator_.feature_importances_)

    # Get the column names (minus the PetID which wasn't in the training/testing sets)
    tmp = df.drop('PetID', axis=1)

    # Plot the first 50 most important features
    fig, ax = plt.subplots(figsize=(20,20)) 
    matplotlib.rcParams.fontsize = 16
    fontsize = 20
    yy = np.array(tmp.columns[sorted_idxs].tolist())
    xx = np.array(sorted_importances)
    plt.barh(yy,xx/np.sum(xx))
    plt.title('Feature importance(Gain)')
    plt.grid(True)
    plt.xlabel('Normalized Gain ', fontsize = fontsize)
    plt.ylabel('Feature name',  fontsize = fontsize)

#============================ Functions related to Neural network processing for the images ========================== #

def imshow_tensor(inp, title=None):
    """
    Gets a tensor, converts to numpy array, normalizes it and plots as image. 
    
    Returns
    ----------
    none. 
    """
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def GetFeatVecForImage(petID, model, layer_name, scaler, normalize, to_tensor, plot_image, device):
    """
    Gets a PetID, attempts to extract the features of its 1st image (if exists) as an output of a layer of a deep
    neural network. 
    
    Parameters
    ----------
    petID: (string) - the PetID we want to search and convert its first image. 
    model: (torchvision.models) - A neural network model we use for feature extraction
    layer_name: (string) - The layer of the neural network we want to extract the data from
    scaler: (torchvision.transforms.Resize) - Resizing function for an image class (PIL.Image.Image).
    normalize: (torchvision.transforms.Normalize) - Normalizing (reduce mean, divide by standard deviation) for an image class.
    to_tensor: (torchvision.transforms.ToTensor) - Converts a PIL Image or numpy.ndarray to tensor.
    plot_image: (bool) - If 'True', will plot every image. Use this for DEBUG ONLY on a small subset of images 
    device: (string) - Either 'cuda'/'cpu'. If 'cuda', will attempt to use GPU for nn processing, else, will use CPU only. 
    
    Returns
    ----------
    np_embedding: (numpy array) A vector of features extracted from the specified layer of the neural network.
    will be zero if no image / single color channel image found. 
    """
    
    # Open and read FIRST image file
    fullpath = os.getcwd() + '\\Data\\train_images\\' + petID + '-1.jpg'
    try:
        img = Image.open(fullpath)
    except:
        return np.zeros((1,512))
    
    # Debug
    if plot_image == True:
        t_img = Variable(normalize(to_tensor(scaler(img))))
        imshow_tensor(t_img)
    
    # Convert to tensor, with unsqueezing for getting a dimension of (1,3,224,224)
    t_img = scaler(img)
    t_img = to_tensor(t_img)
    
    # I skip handling of B/W pictures... 
    if t_img.shape[0] == 1:
        return np.zeros((1,512))
    
    t_img = Variable(normalize(t_img).unsqueeze(0))
    t_img = t_img.type(torch.FloatTensor)
    t_img = t_img.to(device)
    
    # Create a vector of zeros that will hold our feature vector
    # The 'avgpool' layer has an output size of 512
    embedding = torch.zeros([1, 512, 1, 1])
    
    # Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        embedding.copy_(o.data)
    
    # Attach that function to our selected layer
    layer = model._modules.get(layer_name)
    h = layer.register_forward_hook(copy_data)
    
    # 6. Run the model on our transformed image
    model(t_img)
    
    # 7. Detach our copy function from the layer
    h.remove()
    
    # 8. Return the feature vector
    np_embedding = np.array(embedding[0,:,0,0])
    np_embedding = np_embedding.reshape((1,512))
    
    return np_embedding

def GetImageFeaturesForAllPetIds(df, n_components):
    """
    Gets a dataframe, goes over all PetID's in it, calculates a feature vector based on the images and 
    return a concatenated version of the DF with these features. 
    
    Parameters
    ----------
    df: (Pandas Dataframe) - Dataframe containing the a PetID column. 
    n_components: (int) - The number of required components to get from the image. (MUST BE LESS THAN 512 
                   because we're pulling features from a layer size of 512). NOTICE: I tested the explained
                   variability and even with a single dimension, more than 90% is explained by it. This is a bit 
                   surprising, but i didn't investigate it further. 
  
    Returns
    ----------
    df: (Pandas Dataframe) The original dataframe with the new features concatenated.
    """

    #Init
    if ( torch.cuda.is_available() ):
        device = 'cuda' 
        print('Inference via CUDA')
    else:
        device = 'cpu'
        print('User requested CUDA but not supported. Inference via CPU')
    
    # Load the resnet18 model
    model = models.resnet18(pretrained=True)
    model.to(device)
    layer_name = 'avgpool'
    
    # Define the scaling/normalization/tensor transformations functions. 
    scaler = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()    

    # Go over all petIds, get their features
    X_fit = np.zeros((df.shape[0], 512))
    petIds = df['PetID'].values.tolist()
    for q,petID in enumerate(petIds):
        if np.mod(q,100) == 0:
            print('Processed %d images of %d...' % (q, df.shape[0]))

        X_fit[q,:] = GetFeatVecForImage( petID, model, layer_name, scaler, normalize, to_tensor, False, device)
    
    # Truncate features vector to more reasonable dimension containing most of the variability
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(X_fit)
    X_svd = svd.transform(X_fit)

    # Generate a dataframe from these components
    X_svd = pd.DataFrame(X_svd, columns=['img_v_{}'.format(i) for i in range(n_components)])
    X_svd.head()

    # Attach to our input DF 
    df = pd.concat((df, X_svd), axis=1)
    
    return df

def getSize(filename):
    """Returns the size of file in bytes"""
    st = os.stat(filename)
    return st.st_size

def getDimensions(filename):
    """Returns the width/height of image in pixels"""
    img_size = Image.open(filename).size
    return img_size 

def GetImageMetaData(df):
    """
    Returns a DataFrame containing 9 columns per each petID that has photos. The extra columns are 
    metadata of the photos, namely, the sum, mean, variance of the photos size, width, height.
    
    Parameters
    ----------
    df: (Pandas Dataframe) - Dataframe containing the a PetID column. 
  
    Returns
    ----------
    agg_train_imgs: (Pandas Dataframe) DataFrame containing the metadata of photos.
    """
    
    split_char = '\\'
    
    # Create a local dataframe of petIDs 
    train_df_ids = df[['PetID']]
    
    # Get filename of images - using glob because os.listdir() doesn't return full path
    train_image_files = sorted(glob.glob(os.getcwd()+'\Data\\train_images\*.jpg'))
    
    # Generate a dataframe with the filenames
    train_df_imgs = pd.DataFrame(train_image_files)
    train_df_imgs.columns = ['image_filename']
    
    # Create a new column for "PetID" as extracted from the filename
    train_imgs_pets = train_df_imgs['image_filename'].apply(lambda x: x.split(split_char)[-1].split('-')[0])
    train_df_imgs = train_df_imgs.assign(PetID=train_imgs_pets)

    # Add column with image size (bytes)
    train_df_imgs['image_size'] = train_df_imgs['image_filename'].apply(getSize)
    
    # Add column for image height/width using a temporary column container
    train_df_imgs['temp_size'] = train_df_imgs['image_filename'].apply(getDimensions)
    train_df_imgs['width'] = train_df_imgs['temp_size'].apply(lambda x : x[0])
    train_df_imgs['height'] = train_df_imgs['temp_size'].apply(lambda x : x[1])
    train_df_imgs = train_df_imgs.drop(['temp_size'], axis=1)
    
    # We define aggregates. We're going to split each of the new columns to have a sum/mean/var per petID.
    aggs = {
        'image_size': ['sum', 'mean', 'var'],
        'width': ['sum', 'mean', 'var'],
        'height': ['sum', 'mean', 'var'],
    }

    # Generate an aggregated dataframe. I print its structure for simplicity
    agg_train_imgs = train_df_imgs.groupby('PetID').agg(aggs)
    print('agg_train_imgs after grouping by aggs:')
    print(agg_train_imgs.head())
    
    # Rename columns
    agg_train_imgs.columns = ['image_size_sum', 'image_size_mean', 'image_size_var', 'width_sum', 
     'width_mean', 'width_var','height_sum', 'height_mean','height_var']

    # Reset the index to count from zero
    agg_train_imgs = agg_train_imgs.reset_index()
    
    return agg_train_imgs 