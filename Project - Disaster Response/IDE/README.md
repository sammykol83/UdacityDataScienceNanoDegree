# Project - Disaster response

Displays a web dashboard where you can enter SMS texts people (in a disaster area) 
write and classifies them into one of 36 categories.

## Table of contents
- [Motivation](#motivation)
- [Quick start](#quick-start)
- [Requirements](#requirements)
- [Short inner workings description](#short-inner-workings-description)
- [Machine Learning Issues discussion](#discussion)
- [Licensing](#licensing)
- [Appendix](#appendix)

## Motivation <a name="motivation"></a>
In a disaster there's a lot of chaos. Help organizations divide tasks between them. 
Some are responsible for food, some for water, etc.. How do they know which
people to attend when they have the least capacity of responding? 
They rely on classifying SMS messages in the disaster area to topics such as "food",
"shelter" etc. Then they can respond. 

So, the project has a database of text messages. It was trained upon them using
required NLP and pipeline methods, and presents the user a friendly tool to 
classify new text messages. 

## Quick start <a name="quick-start"></a>
Run the following commands in the project's root directory to set up your 
database and model.
- To run ETL pipeline that cleans data and stores in database
  `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
- To run ML pipeline that trains classifier and saves
  `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
- Depending on the platform you're running from (probably windows/linux) you
should change the following flag:
```run_on_udacity_terminal = 0``` in the ```run.py``` file. If windows host, select
'0', else, select '1'. 
- Run the following command in the app's directory to run your web app: ```python run.py```
* Go to the web page address displayed in the cmd line output. For example:
`Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)`.
Press on the link or copy it to your browser and view the dashboard.  

## Requirements <a name="requirements"></a>
See the ```requirements.txt``` file for all packages (and their versions) 
used to generate this project. 

## Project Motivation <a name="#project-motivation"></a>
The project is a showcase for my "capabilities" of deploying a web dashboard using
several tools and methods, both in the frontend and backend.

## Short inner workings description <a name="short-inner-workings-description"></a>
The structure of the project: 

Backend.
- Dataset from Udacity (given in exercise as *.csv files and transformed to SQL DB).
- Python functions (using pandas, plotly) analyze and display results regarding the data.
- Everything is wrapped by a [flask](http://flask.pocoo.org/). 

FrontEnd:
- HTML page designed via [bootstrap](https://getbootstrap.com/). (Bootstrap hides
from us the javascript and complex HTML required for getting responsive and nice
page). 
- Plotly is used for graphs. 
 
ETL and ML pipelines details:
* ETL pipeline: file ```process_data.py``` performs the following steps:
  * reads 2 csv files. 
  * Merges them upon a common field. 
  * Converts the "categories" field from one of them to "one-hot" encoding.
  * Removes duplicate lines. 
  * Saves into a SQL DB. 
* The ML pipeline: file ```train_classifier.py``` performs the following steps:
  * Loads the data from the SQL DB into a pandas frame. 
  * Splits the data into messages (for training) and categories (for labels). 
  * Performs word tokenization on the messages and lemmatizes them. 
    (e.g. splits to words, each word is converted to it's base form, etc..)
  * Builds a ML pipeline of the following type:
    tokenized messages -->  CountVectorizer --> TfidfTranformer --> 
    MultiOutputClassifr(Random forest), where, as a reminder:
    tokenized messages: list of strings (words)
    
    CountVectorizer: Convert the lists of words to a matrix of token counts. 
    We do not provide an a-priori dictionary so number of features will be equal 
    to the vocabulary size found by analyzing the lists of words.
    
    TfidfTranformer: Converts the matrix in such a way that common words which 
    repeat more have less weight. 
    
    MultiOutputClassifier: A classifer container that actually fits one classifier 
    of some type per target label. In our case, we train a "Random Forest" classifier
    against each one of the 36 labels. 
    Reminder: Random forests are a way of averaging multiple deep decision trees, 
    trained on different parts of the same training set, with the goal of reducing the variance.
    
## Machine Learning Issues discussion <a name="discussion"></a>
The dataset is imbalanced (ie some labels like water have few examples). 
I present a short discussion how this imbalance affects training the model
and my thoughts about emphasizing precision or recall for the various categories.

We begin with the test set report for the classifier I trained. (Metrics per label).
You can observe them at the [appenix](#appendix).
As can be seen, for many labels in our TEST SET we have an imbalance situation. 
For example "missing_people": only 1.1% of test messages were classified under
the tag of "missing people". Reminder: TP = True Positive, TN = True Negative. 

* We remember that accuracy is (TP+TN)/(total) isn't a good metric. We can just 
decide that none of the messages are tagged with missing_people and still get 
more than 98% accuracy. 

* Precision and Recall: Precision = TP/(TP+FP), Recall = TP/(TP+FN).
What's the meaning? If finding true positives is critical, we should focus 
on the Recall metric. It tells us the ratio of true detections out of all the True positives.
Precision tells us out of all the CLASSIFIED as True positives, how many did we classify correctly. 
Thus, we emphasize precision when we can "suffer" the consequences of misclassification. 

* Per category, we should emphasize either the precision or recall scores. I would
do that depending on how "life saving critical" the category is and what's the cost
of misclassifying. 
For example, "water" is critical, or "medical help" while "shops" is not. 
Unfortunately, we see in the appendix that our recall score for the imbalanced 
labels is very bad. 

* This brings us back to the last question of how to train on an imbalanced dataset. 
There are several options: 
  * We can oversample the imbalanced classes, meaning, we extract a new DB with more 
    messages from the less frequent categories. 
  * We can randomly undersample the majority categories. 
  * We could try using different classifiers and metrics. Example: use SVM classifier,
    we change thresholds and plot the ROC curve. We look at the TPR which is the same
    as recall and we choose threhsold that maximize it.

Summary: Current performance isn't good due to the imbalanced data and improving 
results requires further work. 
    
## Licensing <a name="licensing"></a>
License: Free. 
Author: Me :). 

## Appendix <a name="appendix"></a>
Classification report from sample training of model:
Column name: request

              precision    recall  f1-score   support
           0       0.90      0.99      0.94      5451
           1       0.88      0.43      0.58      1103

Column name: offer

              precision   recall  f1-score   support
           0       1.00      1.00      1.00      6526
           1       0.00      0.00      0.00        28

Column name: aid_related

              precision    recall  f1-score   support
           0       0.76      0.88      0.81      3834
           1       0.78      0.60      0.68      2720

Column name: medical_help

              precision    recall  f1-score   support
           0       0.93      1.00      0.96      6050
           1       0.67      0.07      0.13       504

Column name: medical_products

              precision    recall  f1-score   support
           0       0.96      1.00      0.98      6252
           1       0.77      0.06      0.10       302

Column name: search_and_rescue

              precision    recall  f1-score   support
           0       0.97      1.00      0.98      6352
           1       0.90      0.04      0.08       202

Column name: security

              precision    recall  f1-score   support
           0       0.98      1.00      0.99      6425
           1       0.50      0.01      0.02       129

Column name: military

              precision    recall  f1-score   support
           0       0.97      1.00      0.98      6338
           1       0.50      0.03      0.06       216
		   
Column name: child_alone

              precision    recall  f1-score   support
           0       1.00      1.00      1.00      6554

Column name: water

              precision    recall  f1-score   support
           0       0.95      1.00      0.97      6151
           1       0.90      0.23      0.36       403

Column name: food

              precision    recall  f1-score   support
           0       0.94      0.99      0.96      5856
           1       0.85      0.47      0.61       698

Column name: shelter

              precision    recall  f1-score   support
           0       0.94      1.00      0.97      5989
           1       0.89      0.32      0.47       565

Column name: clothing

              precision    recall  f1-score   support
           0       0.99      1.00      0.99      6453
           1       0.50      0.05      0.09       101

Column name: money

              precision    recall  f1-score   support
           0       0.98      1.00      0.99      6406
           1       0.71      0.03      0.06       148

Column name: missing_people

              precision    recall  f1-score   support
           0       0.99      1.00      0.99      6483
           1       0.00      0.00      0.00        71

Column name: refugees

              precision    recall  f1-score   support
           0       0.97      1.00      0.98      6345
           1       0.50      0.01      0.02       209

Column name: death

              precision    recall  f1-score   support
           0       0.96      1.00      0.98      6252
           1       0.85      0.14      0.23       302

Column name: other_aid

              precision    recall  f1-score   support
           0       0.87      1.00      0.93      5677
           1       0.60      0.01      0.03       877

Column name: infrastructure_related

              precision    recall  f1-score   support
           0       0.94      1.00      0.97      6141
           1       0.25      0.00      0.00       413

Column name: transport

              precision    recall  f1-score   support
           0       0.96      1.00      0.98      6255
           1       0.88      0.05      0.09       299

Column name: buildings

              precision    recall  f1-score   support
           0       0.95      1.00      0.98      6225
           1       0.76      0.10      0.17       329

Column name: electricity
              precision    recall  f1-score   support

           0       0.98      1.00      0.99      6445
           1       0.86      0.06      0.10       109

Column name: tools

              precision    recall  f1-score   support
           0       0.99      1.00      1.00      6513
           1       0.00      0.00      0.00        41

Column name: hospitals

              precision    recall  f1-score   support
           0       0.99      1.00      0.99      6479
           1       0.00      0.00      0.00        75

Column name: shops

              precision    recall  f1-score   support
           0       1.00      1.00      1.00      6522
           1       0.00      0.00      0.00        32

Column name: aid_centers

              precision    recall  f1-score   support
           0       0.99      1.00      0.99      6484
           1       0.00      0.00      0.00        70

Column name: other_infrastructure

              precision    recall  f1-score   support
           0       0.96      1.00      0.98      6280
           1       0.00      0.00      0.00       274

Column name: weather_related

              precision    recall  f1-score   support
           0       0.87      0.97      0.91      4698
           1       0.88      0.62      0.73      1856

Column name: floods

              precision    recall  f1-score   support
           0       0.95      1.00      0.97      6030
           1       0.89      0.38      0.53       524

Column name: storm

              precision    recall  f1-score   support
           0       0.94      0.99      0.97      5950
           1       0.80      0.43      0.56       604

Column name: fire

              precision    recall  f1-score   support
           0       0.99      1.00      0.99      6485
           1       1.00      0.01      0.03        69

Column name: earthquake

              precision    recall  f1-score   support
           0       0.97      0.99      0.98      5902
           1       0.91      0.72      0.80       652

Column name: cold

              precision    recall  f1-score   support
           0       0.98      1.00      0.99      6418
           1       0.67      0.06      0.11       136

Column name: other_weather

              precision    recall  f1-score   support
           0       0.95      1.00      0.97      6223
           1       0.50      0.02      0.03       331

Column name: direct_report

              precision    recall  f1-score   support
           0       0.87      0.98      0.92      5296
           1       0.85      0.35      0.50      1258