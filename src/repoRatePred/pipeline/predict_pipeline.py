from pathlib import Path
from src.repoRatePred.config.configuration import ConfigurationManager
from src.repoRatePred.logger import logger
from src.repoRatePred.utils.common import load_object
import pickle
import pandas as pd

class PredictPipeline:
        
    def predict(self, features):
        config = ConfigurationManager().get_prediction_config()
        loaded_preprocessor = load_object(Path(config.preprocessor_path))
        loaded_model = load_object(Path(config.model_path))
        
        features = self.get_data_as_dataframe(features)
        preprocessed_features = loaded_preprocessor.transform(features)
        pred = loaded_model.predict(preprocessed_features)
        return pred

    def get_data_as_dataframe(self, data):
        try:
            dataframe = {
                'GoldPrice': data['GoldPrice'],
                'BankRate': data['BankRate'],
                'CrudeOilRate': data['CrudeOilRate'],
                'UsdInr': data['UsdInr'],
                'PreviousRepoRate': data['PreviousRepoRate']
            }
            df = pd.DataFrame(dataframe, index=[0])
            logger.info(df)
            return df
        except Exception as e:
            logger.info(f'Exception Occured in prediction pipeline, ERROR: {e}')