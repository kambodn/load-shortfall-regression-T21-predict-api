"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json
from sklearn import preprocessing

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    df = feature_vector_df
    df = df.drop(['Unnamed: 0', "Barcelona_temp",
                  'Barcelona_temp_min',
                 'Bilbao_temp',
                  'Bilbao_temp_max',
                  'Madrid_temp',
                  'Madrid_temp_min',
                  'Seville_temp_min',
                  'Valencia_temp',
                  'Valencia_temp_min',
                  'timeDayofyear',
                  'timeElapsed',
                  'timeWeek',
                  'timeYear', 
                  'Madrid_weather_id', 
                  'Barcelona_weather_id', 
                  'Seville_weather_id', 
                  'Bilbao_weather_id'], axis=1)
    names = df.columns
    #Extract only numbers out of Seville_pressure and Valencia_wind_deg columns
    df['Seville_pressure'] = df['Seville_pressure'].str.extract('(\d+)')
    df['Seville_pressure'] = pd.to_numeric(df['Seville_pressure'])
    df['Valencia_wind_deg'] = df['Valencia_wind_deg'].str.extract('(\d+)')
    df['Valencia_wind_deg'] = pd.to_numeric(df['Valencia_wind_deg'])
    # fill Valencia_pressure NAN values with mode
    df['Valencia_pressure'] = df['Valencia_pressure'].fillna(1016.0)
    # fill NAN values with mode
    #for i in df.columns:
    #    if df[i].isnull().sum()>0:
    #        df[i]=df[i].fillna(df[i].mode()[0])
    # create scaler object
    scaler = preprocessing.StandardScaler()
    # create scaled version of the predictors
    X_scaled = scaler.fit_transform(df)
    predict_vector= pd.DataFrame(X_scaled, columns=names)
    
    # ------------------------------------------------------------------------
    
    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
