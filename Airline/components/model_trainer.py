from Airline.utils.utils import *
import optuna
from Airline.entity import config_entity
from Airline.entity import artifact_entity
from Airline.exception import AirlineException
from Airline.logger import logging
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from Airline.entity.config_entity import SavedModelConfig
from Airline.constants import *
import mlflow
import optuna
from Airline.logger import *
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score , confusion_matrix ,roc_auc_score ,precision_score , recall_score , f1_score

class Experiments_evaluation:
    def __init__(self,experiment_name, run_name):
        self.experiment_name = experiment_name
        self.run_name = run_name
        
        self.best_model_run_id =None
        self.best_model_uri =None
        self.model_path=None
        self.artifact_uri = None
        self.model_name=None
        
    def get_best_model_run_id(self, experiment_name, metric_name):
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        runs = mlflow.search_runs(experiment_ids=[experiment_id], filter_string='', order_by=[f"metrics.{metric_name} DESC"])
        
        if runs.empty:
            print("No runs found for Specific Experiment and metric")
            return None
         # Get the best run
        best_run = runs.iloc[0]
        self.best_model_run_id = best_run.run_id
        
        # Load the best model
        self.best_model_uri=(f"runs:/{self.best_model_run_id}/model")
    
    # download model from artifacts directory
    def download_model(self, dst_path):
        self.artifact_uri=mlflow.get_run(self.best_model_run_id).info.artifact_uri
        model_uri = f"{self.artifact_uri}/{self.model_name}"
        model = mlflow.pyfunc.load_model(model_uri)
        save_object(file_path=dst_path,obj=model)
# to create report
    def create_run_report(self):
        # Create an MLflow client
        client = MlflowClient()
        run_id=self.best_model_run_id
        # Get the run details
        run = client.get_run(run_id)

        # Report Data 
        # List the contents of the artifact_uri directory
        model_name = self.model_name
        parameters = run.data.params
        metrics = str(run.data.metrics['accuracy_score'])  # Retrieve metrics

        return model_name,parameters,metrics
    
    def run_mlflow_experiment(self,accuracy_score,model,parameters,model_name):
        
        self.model_name=model_name
        # Create or get the experiment
        mlflow.set_experiment(self.experiment_name)
        
        # Start a run
        with mlflow.start_run(run_name=self.run_name):
            # Log metrics, params, and model
            mlflow.log_metric("accuracy_score", float(accuracy_score))
            mlflow.log_params(parameters)
            mlflow.sklearn.log_model(model, f"{model_name}")

        
        logging.info("Checking for best model from the Mlflow Logs")
        
        self.get_best_model_run_id(metric_name='accuracy_score', experiment_name=self.experiment_name)
        
        print(f"Best model Run id: {self.best_model_run_id}")

        return self.best_model_run_id   

class OptunaTuner:
    def __init__(self, model, params, X_train, y_train, X_test, y_test):
        self.model = model
        self.params = params
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def Objective(self, trial):
        param_values = {}
        for key, value_range in self.params.items():
          
            if value_range[0] <= value_range[1]:
                if isinstance(value_range[0], int) and isinstance(value_range[1], int):
                    param_values[key] = trial.suggest_int(key, value_range[0], value_range[1])
                else:
                    param_values[key] = trial.suggest_float(key, value_range[0], value_range[1])
            else:
                raise ValueError(f"Invalid range for {key}: low={value_range[0]}, high={value_range[1]}")

        self.model.set_params(**param_values)

        # Fit the model on the training data
        self.model.fit(self.X_train, self.y_train)

        # Predict on the test data
        y_pred = self.model.predict(self.X_test)

        # Calculate the R2 score as the objective (maximize R2)
        r2 = accuracy_score(self.y_test, y_pred)

        return r2

    def tune(self, n_trials=100):
        study = optuna.create_study(direction="maximize")  # maximize R2 score
        study.optimize(self.Objective, n_trials=n_trials)

        best_params = study.best_params
        print(f"Best parameters: {best_params}")

        # Set the best parameters to the model
        self.model.set_params(**best_params)

        # Retrain the model with the best parameters on the entire training set
        self.model.fit(self.X_train, self.y_train)

        # Evaluate the model on the test set using R2 score
        y_pred_test = self.model.predict(self.X_test)
        best_ac_score = accuracy_score(self.y_test, y_pred_test)
        print(f"Best R2 Score on Test Set: {best_ac_score}")


        # Here, we return the tuned model and the best R2 score on the test set
        return best_ac_score, self.model, best_params

class trainer:
    def __init__(self) -> None:
        self.model_dict={
                           "Random_Forest_Classification": RandomForestClassifier(),
                        
                        }
        
        self.param_dict = {
                            "Random_Forest_Classification": {
                                "n_estimators": [100, 300],
                                "max_depth": [1, 50],
                                "min_samples_split": [2, 20],
                                "min_samples_leaf": [1, 14],
                                'random_state': [1, 100]
                            }
                        }
        
class ModelTrainer :

    def __init__(self,model_training_config:config_entity.ModelTrainingConfig,
                    data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        try:
            self.model_training_config=model_training_config
            self.data_transformation_artifact=data_transformation_artifact
            
            # Accessing config file paths 
            self.trained_model_path=self.model_training_config.model_object_file_path
            self.trained_model_report=self.model_training_config.model_report
            
            self.saved_model_config=SavedModelConfig()
            self.saved_model_dir=self.saved_model_config.saved_model_dir
            
        except Exception as e:
            raise AirlineException(e, sys)
    def convert_parameters(self,parameters):

        #mlflow Donwload paramd data in the string format 

        # Convert parameters to the appropriate data types
        converted_params = {}
        for key, value in parameters.items():
            try:
                # Try to convert the value to an integer
                converted_value = int(value)
            except ValueError:
                try:
                    # If the conversion to int fails, try to convert it to a float
                    converted_value = float(value)
                except ValueError:
                    # If both int and float conversions fail, keep it as a string
                    converted_value = value

            converted_params[key] = converted_value

        return converted_params

    def model_selection(self, X_train, y_train, X_test, y_test,params_dict,models_dict):

        logging.info("Parameter Tuning ...")
        results = {}  # Dictionary to store the best models and their AUC scores
        tuned_models = []  # List to store the tuned models

        for model_name, model in models_dict.items():
            logging.info(f"Tuning and fitting model ----------->>>>  {model_name}")
            tuner = OptunaTuner(model, params=params_dict[model_name], X_train=X_train, y_train=y_train.ravel(),
                                X_test=X_test, y_test=y_test.ravel())

            best_r2_score, tuned_model, best_params = tuner.tune(n_trials=5)

            logging.info(f"Best R2 score for {model_name}: {str(best_r2_score)}")
            logging.info(f"Best Parameters for {model_name}: {best_params}")
            logging.info("----------------------")

            tuned_models.append((model_name, tuned_model, best_params))
            results[model_name] = best_r2_score

        result_df = pd.DataFrame(results.items(), columns=['Model', 'accuracy_score'])
        logging.info(f"Prediction Done: {result_df}")

        result_df_sorted = result_df.sort_values(by='accuracy_score', ascending=False)
        best_model_name = result_df_sorted.iloc[0]['Model']

        for model_tuple in tuned_models:
            if model_tuple[0] == best_model_name:
                best_model_name=model_tuple[0]
                best_model = model_tuple[1]
                best_r2_score = result_df_sorted.iloc[0]['accuracy_score']
                best_params = model_tuple[2]
                break

        return  best_model_name,best_model, best_r2_score, best_params
        # fro mlflow create its model
    def fit_and_evaluate_model(self,model_dict, param_dict, X_train, y_train, X_test, y_test):
        accuracy_score = {}
        
                
        for model_name, model_object in model_dict.items():
            if model_name in param_dict:

                logging.info(f"Fitting the Paramters in the Model : {model_name}")

                parameters = param_dict[model_name]

                converted_params=self.convert_parameters(parameters)
                model_object.set_params(**converted_params)
            
            model_object.fit(X_train, y_train)
            y_pred = model_object.predict(X_test)
            r2 = accuracy_score(y_test, y_pred)

            print(f"Accuracy_score : {r2}")
           
            
        return str(r2),model_object
        
    def start_model_training(self):
        
        try:
            X_train=load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            X_test=load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            y_train=load_numpy_array_data(file_path=self.data_transformation_artifact.train_target_file_path)
            y_test=load_numpy_array_data(file_path=self.data_transformation_artifact.test_target_file_path)
            
            logging.info(f" Shape of the data X_train : {X_train.shape}   y_train : {y_train.shape}    X_test : {X_test.shape}   y_test : {y_test.shape}")
            
            params_file = 'params.yaml'


            if not os.path.exists(params_file):
                # Access Model Dictionary Class
                models=trainer()
                # Create the HyperparameterTuner instance and tune hyperparameters
                logging.info(" Parameter Tunning .....")

                best_model_name,best_model, accuracy_score, best_params=self.model_selection(X_test=X_test,X_train=X_train,
                                                                                            y_test=y_test.ravel(),y_train=y_train.ravel(),
                                                                                 params_dict=models.param_dict,models_dict=models.model_dict)
            
                print(best_model)
                print('===')
                print(best_model_name)
            
                # Model Report 
                model_report={
                    "Experiment": self.model_training_config.mlflow_experiment,
                    "Run_name" : self.model_training_config.mlflow_run_id,
                    "Model_name": best_model_name,
                    "accuracy_score": str(accuracy_score),
                    "Parameters": best_params
                }


                file_path=self.model_training_config.model_object_file_path
                save_object(file_path=file_path, obj=best_model)
                
                # Save the report as a YAML file
                file_path = os.path.join(self.model_training_config.model_report)
                with open(file_path, 'w') as file:
                    yaml.dump(model_report, file)
                logging.info("Model  and report saved to Artifact Folder.")

                # Save the report as a params.yaml
                file_path = os.path.join('params.yaml')
                with open(file_path, 'w') as file:
                    yaml.dump(model_report, file)
                logging.info("Params.yaml file saved to the directory.")



            else: 

                logging.info("Reading params.yaml file from the Directory")
                model_dict={}
                param_dict={}
                models=trainer()

                param_yaml_data=read_yaml_file('params.yaml')
                logging.info("---------- Params.Yaml Data ---------- ")

                param_model_name=param_yaml_data['Model_name']
                parameters=param_yaml_data['Parameters']

                logging.info(f" Model_name : {param_model_name}")
                logging.info(f"Parameters : {parameters}")


                logging.info("---------------------------------------")

                for model_name, model in models.model_dict.items():
                    if model_name==param_model_name:
                        model_dict={param_model_name : model}
                        break
                
                param_dict={ param_model_name : parameters}

                logging.info(f" Input for training Models Dictionary : {model_dict} Parameters Dictionary : {param_dict}")

                accuracy_score,fitted_model=self.fit_and_evaluate_model(X_train=X_train,
                                                                X_test=X_test,
                                                                y_train=y_train.ravel(),
                                                                y_test=y_test.ravel(),
                                                                model_dict=model_dict,param_dict=param_dict)


                best_params=parameters
                best_model_name=param_model_name
                accuracy_score=accuracy_score
                experiment=param_yaml_data['Experiment']
                run_id=param_yaml_data['Run_name']

                # Model Report 
                model_report={
                "Experiment": experiment,
                "Run_name" : run_id,
                "accuracy_score": str(accuracy_score),
                "Parameters": best_params,
                "Model_name":best_model_name
                }

                logging.info(" Saving Model And report ")

                file_path=self.model_training_config.model_object_file_path
                save_object(file_path=file_path, obj=fitted_model)

                # Save the report as a YAML file
                file_path = os.path.join(self.model_training_config.model_report)
                with open(file_path, 'w') as file:
                    yaml.dump(model_report, file)
                logging.info("Model  and report saved to Artifact Folder.")



            
            logging.info(f"-------------")
            os.makedirs(self.saved_model_dir,exist_ok=True)
            # Check if saved_model_directory consits of any contents 
            contents = os.listdir(self.saved_model_dir)

            if not contents:
                # Saving Model object and report 
                logging.info(f"Model Report: {model_report}")
                
                file_path=os.path.join(self.saved_model_dir,'model.pkl')
                save_object(file_path=file_path, obj=best_model)
                logging.info("Model saved.")
                
                # Save the report as a YAML file
                file_path = os.path.join(self.saved_model_dir, 'report.yaml')
                with open(file_path, 'w') as file:
                    yaml.dump(model_report, file)

                # Save the report as a params.yaml
                file_path = os.path.join('params.yaml')
                with open(file_path, 'w') as file:
                    yaml.dump(model_report, file)
                logging.info("Model  and report saved to Artifact Folder.")

                logging.info("Report saved as YAML file.")
            
            else:
                # Saved Model Data Exits in the Ditrectory
                report_file_path=os.path.join(self.saved_model_dir,'report.yaml')
                saved_model_report_data = read_yaml_file(file_path=report_file_path)
                
                # Model Trained Artifact Data
                artifact_model_score=float(accuracy_score)
                saved_model_score=float(saved_model_report_data['accuracy_score'])
                
                if artifact_model_score > saved_model_score:
                    # Model Report 
                    
                    logging.info(" Artifact Model is better than the Saved model ")
                    
                    model_report={
                        "Experiment": self.model_training_config.mlflow_experiment,
                        "Run_name" : self.model_training_config.mlflow_run_id,
                        "Model_name": best_model_name,
                        "accuracy_score": str(accuracy_score),
                        "Parameters": best_params
                    }

                    
                    logging.info(f"Model Report: {model_report}")
                    
                    file_path=os.path.join(self.saved_model_dir,'model.pkl')
                    save_object(file_path=file_path, obj=best_model)
                    logging.info("Model saved.")
                    
                    # Save the report as a YAML file
                    file_path = os.path.join(self.saved_model_dir,'report.yaml')
                    with open(file_path, 'w') as file:
                        yaml.dump(model_report, file)

                    logging.info("Report saved as YAML file.")
                else:

                    saved_report=os.path.join(self.saved_model_config.saved_model_dir,'report.yaml')
                    saved_report_data=read_yaml_file(saved_report)
                    with open('params.yaml', 'w') as file:
                        yaml.dump(saved_report_data, file)

                    logging.info(" Saved Model in the Directory is better than the Trained Model ")
                


            param_data=read_yaml_file('params.yaml')
            experiment_name=param_data['Experiment']
            run_name=param_data['Run_name']
            accuracy_score=param_data['accuracy_score']
            parameters=param_data['Parameters']
            param_model_name=param_data["Model_name"]

            # Report 
            saved_report_file_path=os.path.join(self.saved_model_config.saved_model_dir,'report.yaml')
            report_data = read_yaml_file(file_path=saved_report_file_path)
            model=load_object(os.path.join(self.saved_model_dir,'model.pkl'))


            # Mlflow Code 
            logging.info(" Mlflow ...")
            logging.info(f"Experiment : {experiment_name} , Run_name: {run_name}")
            print(f"Experiment : {experiment_name} , Run_name: {run_name}")
            Exp_eval=Experiments_evaluation(experiment_name=experiment_name,run_name=run_name)

            # Getting best model from the Mlflow log and Saving in the Directory 
            Exp_eval.run_mlflow_experiment(accuracy_score=accuracy_score,model=model,parameters=parameters,model_name=param_model_name)    
            
            # Model_path 
            saved_model_path=os.path.join(self.saved_model_dir,'model.pkl')          
            Exp_eval.download_model(dst_path=saved_model_path)     

           
            model_name,parameters,metrics=Exp_eval.create_run_report()

            model_report={
                "Experiment": experiment_name,
                "Run_name":run_name,
                "Accuracy_Score":metrics,
                "Model_name":model_name,
                "Parameters":parameters
            }

            saved_report_path=os.path.join(self.saved_model_config.saved_model_dir,'report.yaml')
            with open(saved_report_path, 'w') as file:
                    yaml.dump(model_report, file)
            logging.info("Report Created ")
            
            file_path='params.yaml'
            with open(file_path, 'w') as file:
                yaml.dump(model_report, file)


            model_trainer_artifact=artifact_entity.ModelTrainerArtifact(trained_model_file_path=self.model_training_config.model_object_file_path,
                                                                            model_artifact_report=self.model_training_config.model_report)
                
                
                
                
            return model_trainer_artifact
        except Exception as e:
            raise AirlineException(e, sys)
            