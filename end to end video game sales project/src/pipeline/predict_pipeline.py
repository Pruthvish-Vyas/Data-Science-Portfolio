import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            logging.info("Loaded model and preprocessor")
            logging.info(f"Features before preprocessing: {features.shape}")
            
            # Get the column names that the preprocessor expects
            numerical_columns = ['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
            categorical_columns = ['Name', 'Platform', 'Genre', 'Publisher']
            
            # Filter features to only include columns that the preprocessor expects
            features_for_transform = features[
                [col for col in numerical_columns + categorical_columns if col in features.columns]
            ]
            
            logging.info(f"Features for transform: {features_for_transform.shape}")
            
            data_scaled = preprocessor.transform(features_for_transform)
            
            # Convert sparse matrix to dense if needed
            if hasattr(data_scaled, "toarray"):
                data_scaled = data_scaled.toarray()
                
            logging.info(f"Scaled data shape: {data_scaled.shape}")
            
            preds = model.predict(data_scaled)
            logging.info(f"Prediction completed: {preds}")
            
            return preds
            
        except Exception as e:
            logging.error(f"Exception occurred during prediction: {str(e)}")
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        Name: str,
        Platform: str,
        Genre: str,
        Publisher: str,
        Year: int,
        NA_Sales: float,
        EU_Sales: float,
        JP_Sales: float,
        Other_Sales: float
    ):
        self.Name = Name
        self.Platform = Platform
        self.Genre = Genre
        self.Publisher = Publisher
        self.Year = Year
        self.NA_Sales = NA_Sales
        self.EU_Sales = EU_Sales
        self.JP_Sales = JP_Sales
        self.Other_Sales = Other_Sales

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Name": [self.Name],
                "Platform": [self.Platform],
                "Genre": [self.Genre],
                "Publisher": [self.Publisher],
                "Year": [self.Year],
                "NA_Sales": [self.NA_Sales],
                "EU_Sales": [self.EU_Sales],
                "JP_Sales": [self.JP_Sales],
                "Other_Sales": [self.Other_Sales]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            logging.error(f"Exception occurred in CustomData: {str(e)}")
            raise CustomException(e, sys)
