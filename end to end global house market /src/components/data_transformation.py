import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            # First, let's print the columns to debug
            logging.info("Identifying columns for transformation")
            
            numerical_columns = [
                'Year',
                'Rent Index',
                'Affordability Ratio',
                'Mortgage Rate (%)',
                'Inflation Rate (%)',
                'GDP Growth (%)',
                'Population Growth (%)',
                'Urbanization Rate (%)',
                'Construction Index'
            ]
            categorical_columns = ['Country']

            num_pipeline= Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            
            # Print columns to debug
            logging.info(f"Train DataFrame Columns: {train_df.columns.tolist()}")
            logging.info(f"Test DataFrame Columns: {test_df.columns.tolist()}")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            # Target column - using 'House Price Index' as the target
            target_column_name = "House Price Index"
            
            # Check if the target column exists
            if target_column_name not in train_df.columns:
                logging.error(f"Target column '{target_column_name}' not found in dataset!")
                logging.info(f"Available columns: {train_df.columns.tolist()}")
                raise CustomException(f"Target column '{target_column_name}' not found in dataset!", sys)
            
            numerical_columns = [
                'Year',
                'Rent Index',
                'Affordability Ratio',
                'Mortgage Rate (%)',
                'Inflation Rate (%)',
                'GDP Growth (%)',
                'Population Growth (%)',
                'Urbanization Rate (%)',
                'Construction Index'
            ]

            input_feature_train_df=train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
