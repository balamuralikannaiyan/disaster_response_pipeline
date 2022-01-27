# Disaster Response Pipeline Project

**Table of Contents**
1. Installation
2. Project Motivation
3. Project Descriptions
4. Files Descriptions
5. Instructions

**Installation**
Below are the libraries used. They are available in Python Anaconda distribution.<br>

pandas<br>
re<br>
sys<br>
json<br>
sklearn<br>
nltk<br>
sqlalchemy<br>
pickle<br>
Flask<br>
plotly<br>
sqlite3<br>


**Project Motivation**
The project aims at building a classifier which classifies the disaster messages into 36 categories. It has a web app in which messages can be input which is classified in the backend and the result is displayed in the web app.The model is trained and tested using the previously available data. 

**Project Descriptions**<br>
The project has three componants which are:

1. ETL Pipeline: process_data.py does the below parts:
    Extracts the messages and categories datasets from csv files
    Merges the two datasets using id and creates one dataset
    Cleans the data to encode the categories column
    Stores it in athe cleaned data in a SQLite database
2. ML Pipeline: train_classifier.py 1. ETL Pipeline: process_data.py does the below parts:
    Loads data from the SQLite database stored in previous step
    Creates training and test sets
    Builds machine learning pipeline after processing text
    Tunes hyperparameters using GridSearchCV
    Outputs results on each of the 36 categories on the test set
    Saves the model built as a pickle file
3. Flask Web App: The web app inputs the message from the user and categorizes the message based on the model in the pickle file.  

**Files Descriptions**<br>
Below is the file structure:

- \app
	- run.py: flask file for web app
	- \templates
		- master.html: main page of app containing visualization
		- go.html: result web page which categorizes the user messages
- \data
		- disaster_categories.csv: categories dataset
		- disaster_messages.csv: messages dataset
		- DisasterResponse.db: disaster response database
		- process_data.py: ETL pipeline 
- \models
		- train_classifier.py: classification model
        

**Instructions**
1. Run the following commands in the project's root directory to set up your database and model.


    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse
        .db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

