# Project - Web Dashboard

Web dashboard for some descriptive statistics for "Google Play" apps.

## Table of contents

- [Quick start](#quick-start)
- [Installation](#installation)
- [Project Motivation](#project-motivation)
- [Short inner workings description](#short-inner-workings-description)
- [File Descriptions](#file-descriptions)
- [Licensing](#licensing)


## Quick start <a name="quick-start"></a>
* Go to the [dashboard](https://google-apps-dashboard.herokuapp.com/) itself.
* Alternatively, run the "myapp.py". Observe the console output. You'll see
the following line: `Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)`.
Press on it and you'll see the webpage. 

## Installation <a name="installation"></a>
The details of python and modules I used for the project is as follows:
- Python 3.6.6 :: Anaconda 4.4.0 (64-bit)

For a list of the rest of the required modules, see the "requirements.txt" file.

## Project Motivation <a name="#project-motivation"></a>
The project is a showcase for my "capabilities" of deploying a web dashboard using
several tools and methods, both in the frontend and backend.

## Short inner workings description <a name="short-inner-workings-description"></a>
The way the project works:
Backend: 
- Dataset from [kaggle](https://www.kaggle.com/lava18/google-play-store-apps).
- Python functions (using pandas, plotly) analyze and display results regarding the data.
- Everything is wrapped by a [flask](http://flask.pocoo.org/). 

FrontEnd:
- HTML page designed via [bootstrap](https://getbootstrap.com/). (Bootstrap hides
from us the javascript and complex HTML required for getting responsive and nice
page). 
- Plotly is used for graphs. 
- Frontend deployed via [Heroku](https://www.heroku.com/home) - a cloud service for 
  deploying apps. 
 
## File Descriptions <a name="file-descriptions"></a>
Within the download you'll find the following directories and files: 
```text
UdacityDataScienceNanoDegree/
+-- Project - Web Dashboard/
    ¦   +-- readme.md
    ¦   +-- myapp.py
    ¦   +-- wrangling_scripts
           ¦  +-- wrangle_data.py
    ¦   +-- myapp
           ¦  +-- __init__.py
           ¦  +-- routes.py
           ¦  +-- templates
                  ¦  +-- index.html
           ¦  +-- static
                  ¦  +-- img
                        ¦  +-- githublogo.png
                        ¦  +-- linkedinlogo.png
    ¦   +-- data
           ¦  +-- googleplaystore.csv
    ¦   +-- web_app
           ¦  +-- data
                 ¦  +-- googleplaystore.csv
           ¦  +-- myapp
                 ¦ ...
           ¦  +-- wrangling_scripts
                 ¦  +-- wrangle_data.py    
    ¦   +-- myapp.py
    ¦   +-- Procfile
    ¦   +-- requirements.txt
    ¦   +-- runtime.txt
```
We separate the folders into 2 parts. All the folders EXCEPT web_app and
the web_app folders. 

The web_app is just a "concentrated" version of the rest of the folders which are
required for [deployment to Heroku](https://medium.com/the-andela-way/deploying-a-python-flask-app-to-heroku-41250bda27d0). 

I'll focus on the file descriptions for the webapp folder:

- `requirements.txt` - List of python packages required for deployment.  
- `runtime.txt` - Python version used for deployment. 
- `Procfile` - Required for deployment by Heroku.
- `myapp.py` - Entry point for flask
- 'wrangle_data.py' - Main processing file. Includes a single function: 
`return_figures()` Which: 
    * Manipulates the data via pandas.
    * Generates plots via Plotly. 
    * Returns everything to the `routes.py`
- `routes.py` - Responsible for:
    * getting the plotly figures. 
    * Convert the figures to JSON for javascript in html template
    * "Render" them for the webpage itself (flask)
    
## Licensing <a name="licensing"></a>
License: Free. 
Author: Me :). 
