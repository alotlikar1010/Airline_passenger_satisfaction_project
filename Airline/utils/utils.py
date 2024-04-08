import pandas as pd
from Airline.logger import logging
from Airline.exception import AirlineException
import os , sys
from Airline.database_config import mongo_client
import yaml
def get_collection_as_dataframe(database_name:str, collection_name: str)->pd.DataFrame:
    
    try:
        
        logging.info(f"Reading data from database: {database_name} and collection: {collection_name}")
        df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        logging.info(f"Found columns: {df.columns}")
        if "_id" in df.columns:
            logging.info(f"Dropping column: _id ")
            df = df.drop("_id",axis=1)
        logging.info(f"Row and columns in df: {df.shape}")
        return df
    except Exception as e:
        logging.info("Error occur in get_collection_as_dataframe function")
        raise AirlineException(e, sys)
    

def read_yaml_file(file_path:str)->dict:
    try:
        with open(file_path,"rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise AirlineException(e,sys) from e
    

def write_yaml_file(file_path,data:dict):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)
        with open(file_path,"w") as file_writer:
            yaml.dump(data,file_writer)
    except Exception as e:
        raise AirlineException(e, sys)