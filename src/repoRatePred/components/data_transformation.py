from src.repoRatePred.entity.config_entity import DataTransformationConfig
from src.repoRatePred.logger import logger
from src.repoRatePred.utils.common import create_directories, save_object

import pandas as pd
from scipy.stats import boxcox
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        create_directories([self.config.root_dir])
        
    def transform_data(self) -> ColumnTransformer:
        LogTransformer = FunctionTransformer(np.log1p)
        #BoxcoxTransformer = FunctionTransformer(boxcox)
        try:
            ## NOTE: Colunm Transformer will change the order of colunms after applying transformation, so check the value of index mentioned in colunmTransformer after eact CT        
            tr1 = ColumnTransformer([
                ("LogTransformer",LogTransformer, [0])
            ], remainder='passthrough')
            
            tr2 = ColumnTransformer([
                ('BoxcoxTransformer', PowerTransformer(method='box-cox'), [2])
            ], remainder='passthrough')
            
            tr3 = ColumnTransformer([
                ('StandardScalar', StandardScaler(), slice(0,4))
            ], remainder='passthrough')
            pipeline = Pipeline(
                steps=[
                    ('tr1', tr1),
                    ('tr2', tr2),
                    ('tr3', tr3)
                ]
            )
            return pipeline
        except Exception as e:
            logger.info(e)
            
    def initiate_data_transformation(self):
        dataset_file_path = self.config.dataset_file_path
        logger.info(f"Reading dataset from {dataset_file_path}")
        df = pd.read_csv(dataset_file_path)
        logger.info(df.head())
        
        logger.info("Droping 1st colunm as it has index values")
        df.drop('Unnamed: 0', axis=1, inplace=True)
        
        logger.info("Changing Dte colunm from object type to DataTime type")
        df['Date'] = pd.to_datetime(df['Date'])
        
        logger.info("Setting Data colunm as Index")
        df.set_index('Date', inplace=True, drop=True)
        
        logger.info("Resampling Daily data into Montly Data")
        df = df.resample('M').last()
        
        logger.info(df.head())
        logger.info(df.shape)
        
        X = df.drop(self.config.targer_colunm, axis=1)
        y = df[self.config.targer_colunm]
        
        logger.info("Adding previous month repoRate as input")
        prev_repoRate = y.shift(1)
        X['PreviousRepoRate'] = prev_repoRate
        X['PreviousRepoRate'][0] = 9

        logger.info(X.head())
        logger.info(X.shape)
        
        X_train, X_test, y_train, y_test  = train_test_split(X,y, test_size=self.config.test_size, random_state=self.config.random_state)
        
        preprocessor = self.transform_data()
        
        logger.info("Preprocessing data")
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        train_dataset = np.c_[X_train_processed, np.array(y_train)]
        test_dataset = np.c_[X_test_processed, np.array(y_test)]
        
        train_dataset = pd.DataFrame(train_dataset)
        logger.info(f"Created train dataset at location {self.config.train_dataset_file_path} with shape {train_dataset.shape}")
        train_dataset.to_csv(self.config.train_dataset_file_path, index=False)
        test_dataset = pd.DataFrame(test_dataset)
        logger.info(f"Created test dataset at location {self.config.test_dataset_file_path} with shape {test_dataset.shape}")
        test_dataset.to_csv(self.config.test_dataset_file_path, index=False)
        
        save_object(self.config.preprocessor_obj_path, preprocessor)
        