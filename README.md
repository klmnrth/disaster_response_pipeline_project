# Disaster Response Pipeline Project

## Project Description
During disasters there is a lot of communication (direct, news, social media..). At the same time the disaster response organizations have least capacity to filter out the messages which are the most important. 
As part of the Data Science Nanodegree program by Udacity in collaboration with Figure Eight the aim of this project is to build a Natural Language Processing tool which is able to categorize messages in order to address the relevant information to the according disaster response organization. The initial dataset provided by Figure Eight contains pre-labelled tweets and messages from real-life disasters.

The Project is splitted into the following parts:

1. Data Processing: The ETL pipeline extracts the data from two csv-files, cleans the data and saves it as a database file.
2. Machine Learning: The Machine Learning pipeline trains and evaluates the model based on the data from the dataset and saves the classifier afterwards.
3. WebApp: The WebApp is able to take any message and categorize it in real-time using the trained classifier.

## Files in the repository
### File structure and explanations

app
| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app
data
|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py
|- InsertDatabaseName.db # database to save clean data to
models
|- train_classifier.py
|- classifier.pkl # saved model
README.md

## Getting Started
### Dependencies
 - Python 3.8.10
 - Machine Learning Libraries: NumPy, SciPy, Pandas, Scikit-Learn
 - Natural Language Process Libraries: NLTK
 - SQLlite Database Libraries: SQLalchemy
 - Web App and Data Visualization: Flask, Plotly


### Installation
Clone this Git Repository.

`git clone https://github.com/klmnrth/disaster_response_pipeline_project`

### Execute the program

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001/

## Additional notebooks
### ETL Pipeline Preparation
In the data folder you can find the ETL Pipeline Perparation Jupyter Notebook which is basically the foundation of the process_data.py script.
### ML Pipeline Preparation
In the models folder you can find the ML Pipeline Perparation Jupyter Notebook which is basically the foundation of the train_classifier.py script.
This notebook can be used to improve your model by trying other Machine Learning algorithms or by tuning the models parameters with GridSearchCV.

## Screenshots

![image](https://user-images.githubusercontent.com/89395929/131359735-66182de4-069d-4805-9516-90ab4bd151aa.png)

![image](https://user-images.githubusercontent.com/89395929/131359807-edc2b623-1f17-43d6-a751-8beb1b9380af.png)
