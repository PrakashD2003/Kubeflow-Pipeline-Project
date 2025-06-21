import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml
from datetime import datetime
import argparse


# Ensure that a directory named 'logs' exist in our root folder (if not it creates one)(for storing log file)
log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

# Logging Configuration
logger = logging.getLogger('Data_Ingestion') # Created Object of logger with name 'Data_Ingetion'
logger.setLevel('DEBUG') # Setting level of logger as 'DEBUG' so that we see debug as well as all other levels after 'debug'

# Creating Handelers
console_handler = logging.StreamHandler() # Console(terminal) handeler
log_file_path = os.path.join(log_dir,'Data_Ingetion_logs.log') # Creating path for log_file
file_handler = logging.FileHandler(log_file_path, encoding="utf-8") # Creates Log file

# Setting Log Levels for Handelers
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
def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        logger.debug("Attempting to load data from: %s", data_url)
        
        df = pd.read_csv(data_url)
       
        logger.info("Data successfully loaded from %s", data_url)
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

# Function for Preprocessing the Dataset
def preprocessing_data (df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        logger.debug("Starting data preprocessing...")
       
        # Removing unnecssary columns
        df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
       
        # Renaming Columns
        df.rename(columns={'v1':'target','v2':'text'},inplace=True)
        
        logger.info('Data PreProcessing Completed')
        return df  # Return the modified DataFrame
    except KeyError as e:
        logger.error('Missing Colunm in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error("Unexpected error occeured while preprocessing: %s", e)
        raise

# Function to save processed train and test dataset
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

        
# Define the main function to execute the data processing pipeline
def main(param_file_path:str, data_url:str, train_output_path: str, test_output_path: str)->str:
    try:
        # Loading Parameters From params.yaml
        params = load_params(param_file_path)
        
        # Set the test dataset size for splitting
        test_size = params['1_Data_Ingestion']['test_size']
        
        # Define the URL of the dataset (CSV file)
        #data_url = "https://raw.githubusercontent.com/PrakashD2003/DATASETS/refs/heads/main/spam.csv"
        
        # Load the dataset from the provided URL
        df = load_data(data_url=data_url)
        
        # Preprocess the dataset (e.g., cleaning, feature extraction, transformation)
        final_df = preprocessing_data(df)
        
        # Split the dataset into training and testing sets
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)
        
        # Save the train and test data to the specified directory
        save_data(train_data, test_data,train_output_path=train_output_path, test_output_path=test_output_path)

    # Handle any unexpected exceptions that may occur during execution
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

# Ensure that the main function runs only when the script is executed directly
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("param_file_path", type=str, help="Path to params.yaml")
    parser.add_argument("data_url", type=str, help="URL to raw data")
    parser.add_argument("train_output_path", type=str, help="Output file path for train.csv")
    parser.add_argument("test_output_path", type=str, help="Output file path for test.csv")
    args = parser.parse_args()

    main(args.param_file_path, args.data_url, args.train_output_path, args.test_output_path)




