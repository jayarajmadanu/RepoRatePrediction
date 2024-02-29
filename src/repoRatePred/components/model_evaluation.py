from src.repoRatePred.logger import logger
from src.repoRatePred.entity.config_entity import ModelEvaluationConfig

import pandas as pd
import mlflow
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

from src.repoRatePred.utils.common import load_object
 
class ModelEvaluation:
    def __init__(self, config:ModelEvaluationConfig):
        self.config = config
        
    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    
    def model_evaluation(self, best_params=None):
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment('Repo-Rate-Prediction')
        mlflow.autolog()
        test_data_path = self.config.test_data_path
        model = load_object(self.config.model_path)
        
        test_df = pd.read_csv(test_data_path)
        X_test = test_df.iloc[:,0:5]
        y_test = test_df.iloc[:,5]
        
        with mlflow.start_run():
            mlflow.sklearn.log_model(model, artifact_path="model")
            predicted_values = model.predict(X_test)
            (rmse, mae, r2) = self.eval_metrics(y_test, predicted_values)
            mlflow.log_metrics({
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            })
            mlflow.log_params(model.get_params())
        logger.info("END")