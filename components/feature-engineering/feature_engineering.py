import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import yaml
from datetime import datetime
import argparse

# Ensure that a directory named 'logs' exist in our root folder (if not it creates one)(for storing log file)
log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

# Logging Configuration
logger = logging.getLogger('Feature_Engineering') # Created Object of logger with name 'Pre_Proccessing'
logger.setLevel("DEBUG") # Setting level of logger as 'DEBUG' so that we see debug as well as all other levels after 'DEBUG'

# Creating Handlers
console_handler = logging.StreamHandler() # Console(terminal) handeler
file_path = os.path.join(log_dir,'Feature_Engineering.log') # Creating path for log_file
file_handler = logging.FileHandler(file_path,encoding="utf-8") # Creates Log file

# Setting Log Levels for Handlers
console_handler.setLevel("DEBUG")
file_handler.setLevel("DEBUG")

# Creating a Formatter and attaching it to handelers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Adding handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


logger.info("\n" + " "*55 + "="*60)
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


# Function to apply TF-IDF transformation to the dataset
# This function converts text data into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency).
# It assigns weights to words based on their importance and transforms the dataset into a numerical format.
def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """Apply TF-IDF transformation to the dataset."""
    try:
        logger.debug('Tranforming text Data using TDIDF...')
        # Validate that the input data contains the required 'text' and 'target' columns
        if 'text' not in train_data.columns or 'text' not in test_data.columns:
            logger.error("Missing 'text' column in input data.")
            raise KeyError("Column 'text' not found in input data.")
        
        if 'target' not in train_data.columns or 'target' not in test_data.columns:
            logger.error("Missing 'target' column in input data.")
            raise KeyError("Column 'target' not found in input data.")
        
        # Ensure max_features is a positive integer
        if not isinstance(max_features, int) or max_features <= 0:
            logger.error("Invalid max_features: %s. It must be a positive integer.", max_features)
            raise ValueError("max_features must be a positive integer.")

        # Initialize the TF-IDF vectorizer
        # max_features determines the number of most important words to keep
        vectorizer = TfidfVectorizer(max_features=max_features)
        
        # Defensive: ensure no NaN in text columns
        train_data['text'] = train_data['text'].fillna("")
        test_data['text'] = test_data['text'].fillna("")

        # Extract the text data (features) and target labels
        X_train = train_data['text'].values  # Training text data
        y_train = train_data['target'].values  # Training labels
        X_test = test_data['text'].values  # Testing text data
        y_test = test_data['target'].values  # Testing labels

        # Fit the vectorizer on the training data and transform it into numerical format
        X_train_tfidf = vectorizer.fit_transform(X_train)  # Learn vocabulary & transform training data
        X_test_tfidf = vectorizer.transform(X_test)  # Transform test data using the same vocabulary

        # Convert the transformed TF-IDF matrices into Pandas DataFrames
        train_df = pd.DataFrame(X_train_tfidf.toarray())  # Convert sparse matrix to DataFrame
        train_df['label'] = y_train  # Add the target labels to the DataFrame

        test_df = pd.DataFrame(X_test_tfidf.toarray())  # Convert sparse matrix to DataFrame
        test_df['label'] = y_test  # Add the target labels to the DataFrame

        # Log success message
        logger.info('TF-IDF applied and data transformed successfully.')
        
        # Return the transformed training and testing datasets
        return train_df, test_df

    except Exception as e:
        # Log and raise any error encountered during processing
        logger.error('Error during TF-IDF transformation: %s', e)
        raise

# Function to save Features Engineered train and test dataset
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, train_output_path: str, test_output_path: str):
    """Save the train and test datasets."""
    try:
        train_output_path = os.path.join(train_output_path, "train.csv")
        test_output_path = os.path.join(test_output_path, "test.csv")

        # Make sure parent directories exist
        os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(test_output_path), exist_ok=True)
        
        logger.info("Saving train and test datasets...")
        train_data.to_csv(train_output_path, index=False)
        test_data.to_csv(test_output_path, index=False)
       
        logger.info('Training and test data saved to: "%s" & "%s" respectively.', train_output_path, test_output_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main(param_file_path:str, train_data_path:str, test_data_path:str, train_output_path: str, test_output_path: str):
    try:
        # Loading Parameters From params.yaml
        params = load_params(param_file_path)

        max_features = params['3_Feature_Engineering']['max_features']
        
        logger.debug("Attempting to load training data from: %s", './data/interim/train_processed.csv')
        train_data = load_data(train_data_path, train_data=True)
       
        logger.debug("Attempting to load testing data from: %s", './data/interim/test_processed.csv')
        test_data = load_data(test_data_path, train_data=False)
        

        train_df, test_df = apply_tfidf(train_data, test_data, max_features)

        save_data(train_df,test_df,train_output_path=train_output_path, test_output_path=test_output_path)
       
    except Exception as e:
        logger.error('Unexpected error occured while the feature engineering process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("param_file_path", type=str, help="Path of the Params.yaml")
    parser.add_argument("train_data_path", type=str, help="Path to load train data CSV")
    parser.add_argument("test_data_path", type=str, help="Path to load test data CSV")
    parser.add_argument("train_output_path", type=str, help="Output file path for train.csv")
    parser.add_argument("test_output_path", type=str, help="Output file path for test.csv")
    args = parser.parse_args()
    main(param_file_path=args.param_file_path, train_data_path=args.train_data_path, test_data_path=args.test_data_path, train_output_path=args.train_output_path, test_output_path=args.test_output_path)
