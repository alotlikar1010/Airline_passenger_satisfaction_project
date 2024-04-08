import os,sys
from Airline.exception import AirlineException
from Airline.logger import logging
from datetime import datetime
from Airline.utils.utils import read_yaml_file
from Airline.constants import *

config_data=read_yaml_file(CONFIG_FILE_PATH)




class TrainingPipelineConfig:
    
    def __init__(self):
        try:
            self.artifact_dir = os.path.join(os.getcwd(),"artifact")
            
            
        except Exception  as e:
            raise AirlineException(e,sys)    


class DataIngestionConfig:
    
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            data_ingestion_key=config_data[DATA_INGESTION_CONFIG_KEY]
            
            
            # self.database_name=config_data[DATA_INGESTION_DATABASE_NAME]
            # self.collection_name=config_data[DATA_INGESTION_COLLECTION_NAME]
            self.database_name="airline"
            self.collection_name="collection"
            
            
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir ,data_ingestion_key[DATA_INGESTION_ARTIFACT_DIR])
            self.raw_data_dir = os.path.join(self.data_ingestion_dir,data_ingestion_key[DATA_INGESTION_RAW_DATA_DIR_KEY])
            self.ingested_data_dir=os.path.join(self.raw_data_dir,data_ingestion_key[DATA_INGESTION_INGESTED_DIR_NAME_KEY])
            self.train_file_path = os.path.join(self.ingested_data_dir,data_ingestion_key[DATA_INGESTION_TRAIN_DIR_KEY])
            self.test_file_path = os.path.join(self.ingested_data_dir,data_ingestion_key[DATA_INGESTION_TEST_DIR_KEY])
            self.test_size = 0.2
        except Exception  as e:
            raise AirlineException(e,sys)    