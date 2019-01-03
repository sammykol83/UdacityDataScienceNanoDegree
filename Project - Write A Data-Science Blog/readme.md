# Project - Write a Data Science Blog Post

## Table of contents

- [Installation](#Installation)
- [Project Motivation](#project-motivation)
- [File Descriptions](#file-descriptions)
- [How To Interact With The Project](#how-to-interact-with-the-project)
- [Licensing, Authors, Acknowledgement](#licensing)



## Installation

The details of python and modules I used for the project is as follows:
- Python 3.6.6 :: Anaconda 4.4.0 (64-bit)
- Numpy 1.15.1
- Pandas 0.23.4
- Matplotlib 2.0.2
- SciKitLearn 0.18.1

Extra library not included in the Anaconda Distibution:
- Seaborn 0.7.1 (https://seaborn.pydata.org/)

The project itself is supplied as a Jupyter notebook. All the utlility functions and dataset required for running the project are supplied in the repository.

## Project Motivation

The project is about "mimicking" or more accurately "semi-mimicking" a Stanford project called ["Deep Solar" ](http://web.stanford.edu/group/deepsolar/home).

The project uses satellite imagery and deep learning methods to find solar panel installations across the USA. It then combines their findings with numerous other data from other databases (such as environmental and demographic databases) to create a new, large database that can be utilized to get insights about the relationships between these factors. (e.g. what's the solar panel deployment density as function of median income, etc..).

First of all, what was interesting is that it's brand new on Kaggle and nobody had the oppurtunity to investigate it yet. 
Also, their published paper was kind a "walkthrough" for me, which is good since it's the first project (for me) I had to analyze some dataset out of the blue. 
Finally, it's a bit different than all the other databases I saw (financial data, housing data, images etc..) so that appealed to me. 

## File Descriptions:
Within the download you'll find the following directories and files: 
```text
UdacityDataScienceNanoDegree/
+-- Project - Write A Data-Science Blog/
    ¦   +-- readme.md
    ¦   +-- features.txt
    ¦   +-- DeepSolar_Utils.py
    ¦   +-- deepsolar_tracts.zip
           ¦  +-- deepsolar_tract.csv
           ¦  +-- deepsolar_tract_modified.csv
    ¦   +-- DeepSolar_ A Machine Learning Framework to Efficiently Construct a Solar Deployment Database in the United States.pdf
    ¦   +-- BeforeAfterDataSetChange.JPG
    ¦   +-- Deep Solar Udacity Analysis.ipynb
```
- "**readme.md"** - This file. 
- "**features.txt**" - The dataset supplied originally in [kaggle] (https://www.kaggle.com/tunguz/deep-solar-dataset/home) doesn't come with an explanation for the columns. I had to "dig" in the authors article to extract the descriptions for all the columns. I did so the best as I can. The result is this file which is supposed to describe most of the data columns in the *.csv files. 
- ***DeepSolar_ A Machine Learning Framework to Efficiently Construct a Solar Deployment Database in the United States.pdf*** - The pdf Article published in "Joule" describing the research and results for the "Deep Solar" project. 
- **DeepSolar_Utils.py** - Python file with utlity functions I use in the Jupyter notebook.
- **Deep Solar Udacity Analysis.ipynb** - Jupyter notebook which performes all the data analysis for the deep solar dataset. 
- **deepsolar_tracts.zip** - This is a zip file that includes 2 database files. Original and modified. (Why modified? See later :))
- **deepsolar_tract.csv** - The original database published in kaggle. Includes various data sources and the solar deployment information. 
- **deepsolar_tract_modified.csv** - The same database with slight modifications needed for "pandas".
- **BeforeAfterDataSetChange.JPG** - An image that shows the **binary** level change i've made to the original csv file. To make a long story short: One of the columns in the database is called "county" (geographical area). One of the counties is called "Dona Ana", however, the charchter**'a'** was actually **'ã'** which can't be read by pandas. So I changed all occurences of that charachter --> Hence, a "modified" csv file.

## How To Interact With The Project
It's fairly easy (I think). 
1. Unzip the **deepsolar_tract_modified.csv** file into your working directory. 
2. Just run the *.ipnyb file. It has all the documentation within it. 

## Licensing, Authors, Acknowledgement
While i've written all the code by myself, credit goes to all of the people involved in the Deep Solar project for their great job and the fact that they released their database into kaggle.
