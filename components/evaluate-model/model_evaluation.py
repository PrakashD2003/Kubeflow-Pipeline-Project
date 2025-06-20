import os
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score
import pickle
import json 
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import yaml
import argparse

# Ensure that a directory named 'logs' exist in our root folder (if not it creates one)(for storing log file)
log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

# Logging Configuration
logger = logging.getLogger('Model_Evaluation')
logger.setLevel('DEBUG')

# Creating Handlers
console_handler = logging.StreamHandler()
file_log_path = os.path.join(log_dir,'Model_Evaluation.log')
file_handler = logging.FileHandler(file_log_path,encoding='utf-8')

# Setting Log Levels for Handlers
console_handler.setLevel('DEBUG')
file_handler.setLevel('DEBUG')

# Creating a Formatter and attaching it to handelers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Adding handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


logger.info("\n" + " "*52 + "="*60)
logger.info(f"NEW RUN STARTED AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("="*60 + "\n")

# Function to Load Parameters from params.yaml
def load_params(param_path:str) ->dict:
    try:
        logger.debug("Loading Params From: %s",param_path)
        with open(param_path,'r') as file:
            params = yaml.safe_load(file)
        logger.info("Params Loaded Successfully From: %s",param_path)
        return params
    except FileNotFoundError:
        logger.debug('File not found: %s',param_path)
        raise
    except yaml.YAMLError as e:
        logger.debug('Yaml error: %s',e)
        raise
    except Exception as e:
        logger.debug('Unexpected error occured while loadind parameters: %s',e)
        raise

# Function for Loadind Trained Model
def load_model(file_path: str) ->RandomForestClassifier:
    """Load a Trained Model From Specified Path"""
    try:
        logger.debug("Loading Model From: %s",file_path)
        with open(file_path,'rb') as file:
            model = pickle.load(file)
        logger.info("Model Loaded Succesfully.")
        return model
    except FileNotFoundError:
        logger.debug("File not Found: %s",file_path)
        raise
    except Exception as e:
        logger.debug("Unexpected Error Occured while Loading the Model: %s",e)
        raise

# Function for loading the Dataset
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        logger.debug("Loading Test Data from: %s",file_path)
        df = pd.read_csv(file_path)
        df.fillna('',inplace=True)
       
        logger.info("Data successfully loaded and NaN filled from %s", file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file: %s", e)
        raise
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except Exception as e:
        logger.error("Unexpected error occeured while loading the data: %s", e)
        raise

# Function to Evaluate the Model
def evaluate_model(clf:RandomForestClassifier,X_test:np.array,Y_test:np.array) ->dict:
    """Evaluate the Model and Returns Evaluation Metrics"""
    try:
        logger.debug("Predicting test data")
        y_test_pred = clf.predict(X_test)
        y_test_proba = clf.predict_proba(X_test)[:,1]
        logger.info("Test Data Predicted Successfully")
        
        logger.debug("Calculating Evalutaion Metics")
        accuracy = accuracy_score(Y_test,y_test_pred)
        precision = precision_score(Y_test,y_test_pred)
        recall = recall_score(Y_test,y_test_pred)
        auc = roc_auc_score(Y_test,y_test_proba)

        metrics_dict = {
            'accuracy':accuracy,
            'precision':precision,
            'recall':recall,
            'auc':auc
        }
        logger.info('Evaluation Metrics Calculated Successfully')
        return metrics_dict
    except Exception as e:
        logger.debug("Unexpected error occured during model evaluation: %s",e)
        raise
# Function to Save the Evaluation Metrics as Json File
def save_metrics(metrics:dict,file_path:str):
    """Saves the Evaluation Metrics to a JSON file"""
    try:
        # Ensure the directory exists before saving the model
        logger.debug("Creating directory for saving Evaluation Metrics...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Creates directories if they don't exist
        logger.info("Successfully Created Directory at: %s", file_path)

        logger.debug('Saving Evaluation Metrics...')
        with open(file_path,'w') as file:
            json.dump(metrics,file,indent=4)
            logger.info('Evaluation Metrics successfully saved to %s', file_path)

    except FileNotFoundError as e:
        # Handles errors in case the specified file path does not exist
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        # Handles any other unexpected errors that might occur
        logger.error('Unexpected error occurred while saving the evaluation metrics: %s', e)
        raise

def main(param_file_path:str, model_load_path:str, test_data_path:str, metrics_save_path:str):
    try:
         # Loading Parameters From params.yaml
        params = load_params(param_file_path)

        # Loading Trained Model
        clf = load_model(model_load_path)
        
        # Loading Test Data
        test_data = load_data(test_data_path)

        # Extract Input(independent) features and targer(dependent) feature from data
        x_test = test_data.iloc[:,:-1]
        y_test = test_data.iloc[:,-1]

        # Calculating Eavluation Metrics
        metrics_dict = evaluate_model(clf,x_test,y_test)
        
        # Saving evaluation metrics as json file
        save_metrics(metrics_dict,metrics_save_path)
    except Exception as e:
        logger.debug("Failed to complete the model evaluation: %s",e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("param_file_path", type=str, required=True, help="Path of the Params.yaml")
    parser.add_argument("train_data_path", type=str, required=True, help="Path to load train data CSV")
    parser.add_argument("test_data_path", type=str, required=True, help="Path to load test data CSV")
    parser.add_argument("metrics_save_path", type=str, required=True, help="Path to save the metrics json")
    
    args = parser.parse_args()
    main(param_file_path=args.param_file_path, test_data_path=args.test_data_path, metrics_save_path=args.metrics_save_path)
