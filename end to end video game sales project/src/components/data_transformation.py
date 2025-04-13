import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

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

    def get_data_transformer_object(self, numerical_columns, categorical_columns):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore')),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f"Train DataFrame Shape: {train_df.shape}")
            logging.info(f"Test DataFrame Shape: {test_df.shape}")
            
            # Define target column
            target_column_name = 'Global_Sales'  # Using Global_Sales as target
            
            # Define expected columns
            expected_numerical_columns = ['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
            expected_categorical_columns = ['Name', 'Platform', 'Genre', 'Publisher']
            
            # Check which columns actually exist in the dataframe
            available_columns = train_df.columns.tolist()
            logging.info(f"Available columns in dataset: {available_columns}")
            
            # Filter to only use columns that exist in the dataframe
            numerical_columns = [col for col in expected_numerical_columns if col in available_columns]
            categorical_columns = [col for col in expected_categorical_columns if col in available_columns]
            
            logging.info(f"Using numerical columns: {numerical_columns}")
            logging.info(f"Using categorical columns: {categorical_columns}")
            
            # Validate that target column exists
            if target_column_name not in train_df.columns or target_column_name not in test_df.columns:
                raise CustomException(f"Target column '{target_column_name}' is missing in the dataset.", sys)

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object(
                numerical_columns=numerical_columns,
                categorical_columns=categorical_columns
            )

            # Prepare input features and target features
            input_feature_train_df = train_df[numerical_columns + categorical_columns]
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df[numerical_columns + categorical_columns]
            target_feature_test_df = test_df[target_column_name]
            
            # Log shapes to debug
            logging.info(f"Input feature train shape: {input_feature_train_df.shape}")
            logging.info(f"Target feature train shape: {target_feature_train_df.shape}")
            logging.info(f"Input feature test shape: {input_feature_test_df.shape}")
            logging.info(f"Target feature test shape: {target_feature_test_df.shape}")

            logging.info("Applying preprocessing object on training and testing dataframes")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            logging.info(f"Transformed input feature train shape: {input_feature_train_arr.shape}")
            logging.info(f"Transformed input feature test shape: {input_feature_test_arr.shape}")

            # Convert sparse matrix to dense if needed
            if hasattr(input_feature_train_arr, "toarray"):
                input_feature_train_arr = input_feature_train_arr.toarray()
            if hasattr(input_feature_test_arr, "toarray"):
                input_feature_test_arr = input_feature_test_arr.toarray()
                
            # Convert target arrays to numpy and reshape to match dimensions
            target_train_arr = np.array(target_feature_train_df).reshape(-1, 1)
            target_test_arr = np.array(target_feature_test_df).reshape(-1, 1)
            
            logging.info(f"Input feature train array shape after conversion: {input_feature_train_arr.shape}")
            logging.info(f"Target train array shape: {target_train_arr.shape}")
            
            # Create the final arrays using np.concatenate
            train_arr = np.concatenate((input_feature_train_arr, target_train_arr), axis=1)
            test_arr = np.concatenate((input_feature_test_arr, target_test_arr), axis=1)
            
            logging.info(f"Final train array shape: {train_arr.shape}")
            logging.info(f"Final test array shape: {test_arr.shape}")

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
            logging.error(f"Exception occurred: {str(e)}")
            # Print more detailed error information
            import traceback
            logging.error(traceback.format_exc())
            raise CustomException(e,sys)
