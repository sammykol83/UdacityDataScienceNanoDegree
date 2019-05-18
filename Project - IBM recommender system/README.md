# Recommendations with IBM

This notebook implements a recommender system on real data from the IBM Watson Studio platform.

## Table of contents
- [Motivation](#motivation)
- [Quick start](#quick-start)
- [Requirements](#requirements)

## Motivation <a name="motivation"></a>
The project analyses the interactions that users have with articles on the IBM Watson Studio platform, and make recommendations to them about new articles they will like. Below you can see an example of what the dashboard could look like displaying articles on the IBM Watson Platform.

![](screen-shot-2018-09-17-at-3.40.30-pm.png)

Though the above dashboard is just showing the newest articles, we can imagine having a recommendation board available here that shows the articles that are most pertinent to a specific user.

In order to determine which articles to show to each user, we perform a study of the data available on the IBM Watson Studio platform, and then follow these steps:
* Exploratory Data Analysis
* Rank Based Recommendations
* User-User Based Collaborative Filtering
* Content Based Recommendations
* Matrix Factorization

## Quick start <a name="quick-start"></a>
Just run the *.ipnyb file in the Jupyter notebook

## Requirements <a name="requirements"></a>
I developed the project on:
* OS: **WINDOWS 10**
* Python ver: **3.7.1** 
* Pandas: 0.23.3
* Numpy: 1.12.1
* Pickle, Matplotlib: Any recent version should work.