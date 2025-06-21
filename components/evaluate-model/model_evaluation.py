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

# Function for Loadind Trained Model
def load_model(model_dir: str) -> RandomForestClassifier:
    """Load a trained model from a directory (expects model.pkl inside)."""
    try:
        model_path = os.path.join(model_dir, "model.pkl")  # or whatever name you saved it as
        logger.debug("Loading Model From: %s", model_path)
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.info("Model Loaded Successfully.")
        return model
    except FileNotFoundError:
        logger.debug("File not found: %s", model_path)
        raise
    except Exception as e:
        logger.debug("Unexpected error while loading the model: %s", e)
        raise

# Function for loading the Dataset
def load_data(input_dir: str, train_data: bool) -> pd.DataFrame:
    """
    Load train or test CSV from a Kubeflow-mounted directory path.

    :param input_dir: Directory path (e.g., train_data.path or test_data.path)
    :param train_data: Flag to determine whether to load 'train.csv' or 'test.csv'
    :return: Loaded DataFrame
    """
    try:
        filename = "train.csv" if train_data else "test.csv"
        file_path = os.path.join(input_dir, filename)

        logger.debug("Attempting to load data from: %s", file_path)
        df = pd.read_csv(file_path)
        logger.info("Data successfully loaded from %s", file_path)
        return df

    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file: %s", e)
        raise
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while loading the data: %s", e)
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
def save_metrics(metrics: dict, output_dir: str):
    """Saves the Evaluation Metrics to a JSON file in a given output directory path from Kubeflow."""
    try:
        # Define full file path
        file_path = os.path.join(output_dir, "metrics.json")

        # Ensure the directory exists
        os.makedirs(output_dir, exist_ok=True)

        logger.debug("Saving evaluation metrics to file: %s", file_path)
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)

        logger.info("Evaluation metrics successfully saved to %s", file_path)

    except Exception as e:
        logger.error("Error while saving metrics: %s",e)

def main(model_load_path:str, test_data_path:str, metrics_save_path:str):
    try:
        
        # Loading Trained Model
        clf = load_model(model_load_path)
        
        # Loading Test Data
        test_data = load_data(test_data_path, train_data=False)

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
    parser.add_argument("model_load_path", type=str, help="Path to load trained Model")
    parser.add_argument("test_data_path", type=str, help="Path to load test data CSV")
    parser.add_argument("metrics_save_path", type=str, help="Path to save the metrics json")
    args = parser.parse_args()

    main(model_load_path=args.model_load_path, test_data_path=args.test_data_path, metrics_save_path=args.metrics_save_path)
