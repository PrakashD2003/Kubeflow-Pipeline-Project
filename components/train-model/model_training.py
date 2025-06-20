import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
import yaml
from datetime import datetime
import argparse

# Ensure that a directory named 'logs' exist in our root folder (if not it creates one)(for storing log file)
log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

# Logging Configuration
logger = logging.getLogger('Model_Training') # Created Object of logger with name 'Pre_Proccessing'
logger.setLevel('DEBUG') # Setting level of logger as 'DEBUG' so that we see debug as well as all other levels after 'DEBUG'

# Creating Handlers
console_handler = logging.StreamHandler() # Console(terminal) handeler
file_handler_path = os.path.join(log_dir,"Model_Training.log") # Creating path for log_file
file_handler = logging.FileHandler(file_handler_path,encoding="utf-8") # Creates Log file

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


logger.info("\n" + " "*50 + "="*60)
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

# Function for loading the Dataset
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        logger.debug("Loading Training Data from: %s",file_path)
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

# Function to train our randomforest model
def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
    """
    Train the RandomForest model.
    
    :param X_train: Training features
    :param y_train: Training labels
    :param params: Dictionary of hyperparameters
    :return: Trained RandomForestClassifier
    """
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in X_train and y_train must be the same.")
        
        logger.debug('Initializing RandomForest model with parameters: %s', params)
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
        
        logger.debug('Model training started with %d samples', X_train.shape[0])
        clf.fit(X_train, y_train)
        logger.info('Model training completed')
        
        return clf
    except ValueError as e:
        logger.error('ValueError during model training: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occured during model training: %s', e)
        raise

# Function to save the trained model
def save_model(model, file_path: str) -> None:
    """
    Save the trained model to a file using pickle.
    
    :param model: Trained model object (e.g., a Scikit-learn model)
    :param file_path: Path where the model should be saved, including the filename and extension (.pkl)
    """
    try:
        # Ensure the directory exists before saving the model
        logger.debug("Creating directory for saving Trained Model...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Creates directories if they don't exist
        logger.info("Successfully Created Directory at: %s", file_path)

        # Open the file in write-binary mode ('wb') to store the model
        logger.debug('Saving Trained Model...')
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)  # Serialize and save the model to the specified file path
        logger.info('Model successfully saved to %s', file_path)

    except FileNotFoundError as e:
        # Handles errors in case the specified file path does not exist
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        # Handles any other unexpected errors that might occur
        logger.error('Unexpected error occurred while saving the model: %s', e)
        raise


# Main function to load data, train the model, and save it
def main(param_file_path:str, train_data_path:str, model_save_path:str):
    try:
        # Loading Parameters From params.yaml
        params = load_params(param_file_path)['4_Model_Training']
        
        # Load preprocessed training data (TF-IDF transformed)
        train_data = load_data(train_data_path)
        
        # Extract input features (X_train) and target labels (y_train) from the dataset
        X_train = train_data.iloc[:, :-1].values  # Select all columns except the last one (features)
        y_train = train_data.iloc[:, -1].values   # Select the last column as target labels
        
        # Train the model using the extracted features and target labels
        clf = train_model(X_train, y_train, params)
        
        # Define the path where the trained model should be saved
        model_save_path = model_save_path
        
        # Save the trained model for future use
        save_model(clf, model_save_path)

    except Exception as e:
        # Log and print an error message if any step fails
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

# Entry point of the script: Execute the main function when the script runs
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("param_file_path", type=str, required=True, help="Path of the Params.yaml")
    parser.add_argument("train_data_path", type=str, required=True, help="Path to load train data CSV")
    parser.add_argument("model_save_path", type=str, required=True, help="Path to save the trained model")
    args = parser.parse_args()
    main(param_file_path=args.param_file_path, train_data_path=args.train_data_path, model_save_path=args.model_save_path)

