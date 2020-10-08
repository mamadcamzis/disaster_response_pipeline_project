# Disaster Response Pipeline Project

## Purpose
The aim of the project is to build an API that classifies disaster messages.
Using web app emergency worker can input a message and get disaster response in several categories. 
For example "water", "shelter", "food",  are kind of possible categories

This is an overview of the homepage


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.
     - I have created a python virtualenv. Activate virtualenv all required packages are already installed.
     Run this command
      `source disasterenv/bin/activate`

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
