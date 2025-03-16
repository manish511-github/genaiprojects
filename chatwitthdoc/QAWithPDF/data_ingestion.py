from llama_index.core import SimpleDirectoryReader
import sys
from Exception import customexception
from Logger import logging

def load_data(data):
    """
    Load PDF documents from a specified directory.
    Parameters:
    - data (str): Path to the directory containing PDF documents.
    Returns:
    - A list of loaded PDF documents. The specific type of documents may vary
    """
    try:
        logging.info("data loaded started...")
        loader = SimpleDirectoryReader("Data")
        documents = loader.load_data()
        logging.info("data loaded successfully")
        return documents
    except Exception as e:
        logging.error("Error occured while loading data")
        raise customexception(e,sys)