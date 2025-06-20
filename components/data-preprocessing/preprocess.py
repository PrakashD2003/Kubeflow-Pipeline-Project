import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
import argparse
from datetime import datetime

nltk.download('stopwords')
nltk.download('punkt')  # Also required for word_tokenize


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

# Function for loading the Dataset
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
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
        logger.error("Unexpected error occeured while loading the data: %s", e)
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
def save_data(train_data_processed: pd.DataFrame,test_data_processed: pd.DataFrame,file_path: str) ->None:
    """Save the proccessed train and test datasets."""
    try:
       # Store the data inside data/interim
        logger.debug("Creating directory for saving processed data...")
        os.makedirs(file_path, exist_ok=True)
        logger.info("Successfully Created Directory at: %s",file_path)

        # Saving Proccessed Train and Test data as CSV inside 'raw' directory
        logger.debug("Saving Processed data to %s", file_path)
        train_data_processed.to_csv(os.path.join(file_path, "train_processed.csv"), index=False)
        test_data_processed.to_csv(os.path.join(file_path, "test_processed.csv"), index=False)
        logger.info('Processed data saved to %s', file_path)
       
        logger.info('Proccessed Training and Test data saved to: %s', file_path)
    except Exception as e:
        logger.error('Unexpected error occeured while saving the data: %s', e)
        raise


def main(train_data_path:str, test_data_path:str, output_path:str, text_column='text', target_column='target'):
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    try:
        # Fetch the data from data/raw
        train_data = load_data(train_data_path)
        test_data = load_data(test_data_path)

        # Transform the data
        logger.debug("Starting DataFrame preprocessing for Training Data...")
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        logger.info(' Training Data Preprocessed Successfully')
        logger.debug("Starting DataFrame preprocessing for Test Data...")
        test_processed_data = preprocess_df(test_data, text_column, target_column)
        logger.info(' Testing Data Preprocessed Successfully')

        # Save data to data/interim
        file_path = output_path
        save_data(train_processed_data,test_processed_data,file_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_path", type=str, required=True, help="Path to load train data CSV")
    parser.add_argument("test_data_path", type=str, required=True, help="Path to load test data CSV")
    parser.add_argument("output_path", type=str, required=True, help="Path to save the output CSV")
    parser.add_argument("text_column", type=str, required=False, help="Name of Text Column to Preprocess")
    parser.add_argument("target_column", type=str, required=False, help="Name of Target Column to Preprocess")
    args = parser.parse_args()
    main(train_data_path=args.train_data_path, test_data_path=args.test_data_path, output_path=args.output_path, text_column=args.text_column, target_column=args.target_column)