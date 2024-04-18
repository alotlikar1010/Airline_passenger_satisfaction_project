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
        
class DataTransformationConfig:
    
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            data_transformation_key = config_data[DATA_TRANSFORMATION_CONFIG_KEY]
            
            self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir , data_transformation_key[DATA_TRANSFORMATION])
            self.transformation_dir = os.path.join(self.data_transformation_dir,data_transformation_key[DATA_TRANSFORMATION_DIR_NAME_KEY])
            self.transformed_train_dir = os.path.join(self.transformation_dir,data_transformation_key[DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY])
            self.transformed_test_dir = os.path.join(self.transformation_dir,data_transformation_key[DATA_TRANSFORMATION_TEST_DIR_NAME_KEY])
            self.preprocessed_dir = os.path.join(self.data_transformation_dir,data_transformation_key[DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY])
            self.feature_engineering_object_file_path =os.path.join(self.preprocessed_dir,data_transformation_key[DATA_TRANSFORMATION_FEA_ENG_FILE_NAME_KEY])
            self.preprocessor_file_object_file_path=os.path.join(self.preprocessed_dir,data_transformation_key[DATA_TRANSFORMATION_PREPROCESSOR_NAME_KEY])
            
        except Exception as e:
            raise AirlineException(e,sys)
        
class ModelTrainingConfig:
    
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        
        model_training_key=config_data[MODEL_TRAINING_CONFIG_KEY]

        
        self.model_training_dir = os.path.join(training_pipeline_config.artifact_dir ,model_training_key[MODEL_TRAINER_ARTIFACT_DIR])
        self.model_object_file_path = os.path.join(self.model_training_dir,model_training_key[MODEL_TRAINER_OBJECT])
        self.model_report =  os.path.join(self.model_training_dir,model_training_key[MODEL_REPORT_FILE])
        
        mlflow_config_key=model_training_key['mlflow']

        self.mlflow_run_id = os.path.join(mlflow_config_key['run_id'])
        self.mlflow_experiment=os.path.join(mlflow_config_key['experiment'])
        

class SavedModelConfig:
    
    def __init__(self):
        saved_model_config_key=config_data[SAVED_MODEL_CONFIG_KEY]
        ROOT_DIR=os.getcwd()
        self.saved_model_dir=os.path.join(ROOT_DIR,saved_model_config_key[SAVED_MODEL_DIR])       
