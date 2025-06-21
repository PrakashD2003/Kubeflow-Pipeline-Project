import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
from nltk.data import path as nltk_data_path
import argparse
from datetime import datetime

# Explicitly tell nltk where to find the data
# Explicitly add path (again for extra safety)
nltk_data_path.append('/usr/share/nltk_data')

# Ensure that a directory named 'logs' exist in our root folder (if not it creates one)(for storing log file)
log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

# Logging Configuration
logger = logging.getLogger("Pre_Processing") # Created Object of logger with name 'Pre_Proccessing'
logger.setLevel("DEBUG") # Setting level of logger as 'DEBUG' so that we see debug as well as all other levels after 'DEBUG'

# Creating Handlers
console_handler = logging.StreamHandler() # Console(terminal) handeler
log_file_path = os.path.join(log_dir,'Pre_Processing_logs.log') # Creating path for log_file
file_handler = logging.FileHandler(log_file_path, encoding="utf-8") # Creates Log file

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


logger.info("\n" + " "*50 + "="*60)
logger.info(f"NEW RUN STARTED AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("="*60 + "\n")

logger.debug(f"NLTK paths: {nltk.data.path}") 

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

# Function to tranform the input text
def transform_text(text: str) ->str:
    """
    Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation, and stemming.
    """
    try:
        # Creating Instance of PorterStemmer
        ps = PorterStemmer()
        # Convert to lowercase
        text = text.lower()
        # Tokenize the text
        text = nltk.word_tokenize(text)
        # Remove non-alphanumeric tokens
        text = [word for word in text if word.isalnum()]
        # Remove stopwords and punctuation
        text = [word for word in text if word not in stopwords.words('english')]
        text = [word for word in text if word not in string.punctuation]
        # Stem the words
        text = [ps.stem(word) for word in text]
        # Join the tokens back into a single string
        return " ".join(text)
    except Exception as e:
        logger.error("Unexpected error occured while transforming text data: %s", e)
        raise

# Function for preprocessing the data
def preprocess_df(df: pd.DataFrame, text_column='text', target_column='target') ->pd.DataFrame:
    """
    Preprocesses the DataFrame by encoding the target column, removing duplicates, and transforming the text column.
    """
    try:
        
        # Encode the target column
        logger.debug('Starting Label Encoding For Target Column...')
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.info('Target column encoded')

        # Remove duplicate rows
        logger.debug('Removing Duplicate Rows...')
        df = df.drop_duplicates(keep='first')
        logger.info('Duplicates removed')
        
        # Apply text transformation to the specified text column
        logger.debug("Starting input text data transformatoin....")
        df[text_column] = df[text_column].apply(transform_text)
        logger.info("Text Data Transformation Completed.")
        return df
    
    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
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
        logger.info(f"Writing train.csv to: {train_output_path}")
        train_data.to_csv(train_output_path, index=False)
        logger.info(f"Writing test.csv to: {test_output_path}")
        test_data.to_csv(test_output_path, index=False)
       
        logger.info('Training and test data saved to: "%s" & "%s" respectively.', train_output_path, test_output_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise



def main(train_data_path:str, test_data_path:str, train_output_path: str, test_output_path: str, text_column='text', target_column='target'):
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    try:
        # Fetch the data from data/raw
        train_data = load_data(train_data_path, train_data=True)
        test_data = load_data(test_data_path, train_data=False)

        # Transform the data
        logger.debug("Starting DataFrame preprocessing for Training Data...")
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        logger.info(' Training Data Preprocessed Successfully')
        logger.debug("Starting DataFrame preprocessing for Test Data...")
        test_processed_data = preprocess_df(test_data, text_column, target_column)
        logger.info(' Testing Data Preprocessed Successfully')

        # Save data 
        save_data(train_data=train_processed_data,test_data=test_processed_data,train_output_path=train_output_path, test_output_path=test_output_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_path", type=str, help="Path to load train data CSV")
    parser.add_argument("test_data_path", type=str, help="Path to load test data CSV")
    parser.add_argument("train_output_path", type=str, help="Output file path for train.csv")
    parser.add_argument("test_output_path", type=str, help="Output file path for test.csv")
    parser.add_argument("text_column", type=str, help="Name of Text Column to Preprocess")
    parser.add_argument("target_column", type=str, help="Name of Target Column to Preprocess")
    args = parser.parse_args()
    main(train_data_path=args.train_data_path, test_data_path=args.test_data_path, train_output_path=args.train_output_path, test_output_path=args.test_output_path, text_column=args.text_column, target_column=args.target_column)