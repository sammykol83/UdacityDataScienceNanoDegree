# Capstone project - Petfinder.my

![](PetFinder.png)
The project is a classifier implementation for the PetFinder.my [Kaggle competition](https://www.kaggle.com/c/petfinder-adoption-prediction/kernels). The goal is to develop a classifier that gets input about a target pet and can estimate how quickly it will be adopted. 

the files in the repository with a small description of each, a summary of the results of the analysis, and necessary acknowledgements.

## Table of contents
- [Motivation](#motivation)
- [Quick start](#quick-start)
- [Summary of results](#results)
- [Requirements](#requirements)
- [Directory structure](#Directory-structure)

## Motivation <a name="motivation"></a>
Millions of stray animals suffer on the streets or are euthanized in shelters every day around the world. If homes can be found for them, many precious lives can be saved — and more happy families created.

PetFinder.my has been Malaysia’s leading animal welfare platform since 2008, with a database of more than 150,000 animals.

Animal adoption rates are strongly correlated to the metadata associated with their online profiles, such as descriptive text and photo characteristics. Thus, the motivation is developing AI tools that will guide shelters and rescuers around the world on improving their pet profiles' appeal, reducing animal suffering and euthanization.

## Quick start <a name="quick-start"></a>
**Data acquisition:**
Part of the data is directly inside the GitHub repo, however, more is required, so there's no choice but to download the entire dataset again, add the missing data files and delete the rest. 
Specifically, you need to add the "train_images.zip" file.

1. Download the data from [here](https://www.kaggle.com/c/10686/download-all)
2. Extract into the "Data" folder on the repo folder on your local machine.
3. Delete the original zip. 
4. Delete all zip files related to test files. 
5. Extract the "train_images.zip" file into the "Data" folder.  
6. Delete the "train_images.zip" file. 

At this point, you should be left with a folder structure described [here](#Directory-structure)

**Runnig the notebook**
1. Open the ```petfinder.ipynb``` Jupyter notebook. 
2. Run cell by cell. 

**NOTICE** File paths are different between Linux/Windows. I used "os.sep" in all places of file access, hopefully, if you're testing on a LINUX environment, all the file openings will succeed. 

## Summary of results <a name="results"></a>
Adoption speed is divided into 5 labels, and thus our domain is a multi-class classification problem. The metric of evaluation is QWK [(Quadratic Weighted Kappa)](https://www.kaggle.com/c/petfinder-adoption-prediction#evaluation).

My solution uses the train data (tabular) + sentiment analysis for text descriptions + features extracted from photos via a ResNet-18 DNN + metadata regarding the photos.

It bunches all into a GBM (implementation via LightGBM) and gave me:

| Training QWK   | CV QWK   | Kaggle QWK/rank in LeaderBoard |
|--------------- | -------- | ------------------------------ |
|    0.71        |  0.37    |     0.308 / (1665 of 2010)     |  

(Though my Kaggle score wasn't derived from my best implementation. Best Public LB is 0.509).

These results reflect:
*  My model was overfitting (Large gap between training and CV results), however, it was the best I could achieve in a reasonable time.
* About 40% of adoption speed estimations were correct. (Over the test data)
* About 50% of the misclassified adoption speeds (which are 60% of the test data results) had an error of "1" unit (Which is less worse than missing by more time units).

More about the results is contained in the *.ipnyb notebook in the code itself...

## Requirements <a name="requirements"></a>
I developed the project on:
* OS: **WINDOWS 10**
* Python ver: **3.7.1** 

See the ```requirements.txt``` file for all packages (and their versions) 
used in this project. 

## Directory structure <a name="directory-structure"></a>
Within the download **AFTER YOU EXTRACT ALL ZIPS (SEE [QUICK START](#quick-start))**: 

```text
UdacityDataScienceNanoDegree/
+-- Project - Capstone - PetFinder.my/
    ¦   +-- readme.md
    ¦   +-- petfinder.ipynb
    ¦   +-- petfinder.html
    ¦   +-- Data
           ¦  +-- breed_labels.csv
           ¦  +-- color_labels.csv
           ¦  +-- state_labels.csv
           ¦  +-- train.csv
           ¦  +-- train_images
                 ¦   +-- 000a290e4-1.jpg
                 ¦   +-- 000a290e4-2.jpg
                 ¦   +-- ...
           ¦  +-- train_sentiment
                 ¦   +-- 000a290e4.json
                 ¦   +-- 000fb9572.json
                 ¦   +-- ...
    ¦   +-- PetFinder.png
    ¦   +-- requirements.txt
    ¦   +-- Utils.py
    ¦   +-- Competition.JPG
```
- "**readme.md"** - This file. 
- "**requirements.txt**" - Contains a list of all python packages and their versions used for this project. 
- **petfinder.ipynb** - Jupyter notebook which performes all the data analysis and modeling for the project
- **petfinder.html** - An HTML copy of the Jupyter notebook in case something goes wrong with running the notebook. 
- **Utils.py** - Python file with utlity functions I use in the Jupyter notebook. This file actually does most of the "heavy lifting" in this project. In contains bulk codes for most of the main parts. The code in the notebook itself is much more shorter and serves for "telling the story" of the data.
- **Data folder** - This folder should contain 4 *.csv files and 2 folders, after all the cleanups described in the "Quick Start" section. The data itself was download from [here](https://www.kaggle.com/c/10686/download-all).
This folder, after extraction, weighs about **1.7Gbyte(!)** of data. The bulk data is due to the large number of photos included. 
- **PetFinder.png** - photo file used for this "readme.md".
- **Competition.jpg** - photo file used for the Jupyter notebook. 
