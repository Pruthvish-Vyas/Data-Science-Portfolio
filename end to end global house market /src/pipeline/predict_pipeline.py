import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object
import os
from src.logger import logging


# You can also add this to configure logging if needed
from src.logger import logging  # If you have a custom logger module


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            
            logging.info("Loading model and preprocessor")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            logging.info("Preprocessing the input features")
            data_scaled = preprocessor.transform(features)
            
            logging.info("Making predictions")
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, 
                 country: str,
                 year: int,
                 rent_index: float,
                 affordability_ratio: float,
                 mortgage_rate: float,
                 inflation_rate: float,
                 gdp_growth: float,
                 population_growth: float,
                 urbanization_rate: float,
                 construction_index: float):
        
        self.country = country
        self.year = year
        self.rent_index = rent_index
        self.affordability_ratio = affordability_ratio
        self.mortgage_rate = mortgage_rate
        self.inflation_rate = inflation_rate
        self.gdp_growth = gdp_growth
        self.population_growth = population_growth
        self.urbanization_rate = urbanization_rate
        self.construction_index = construction_index

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Country": [self.country],
                "Year": [self.year],
                "Rent Index": [self.rent_index],
                "Affordability Ratio": [self.affordability_ratio],
                "Mortgage Rate (%)": [self.mortgage_rate],
                "Inflation Rate (%)": [self.inflation_rate],
                "GDP Growth (%)": [self.gdp_growth],
                "Population Growth (%)": [self.population_growth],
                "Urbanization Rate (%)": [self.urbanization_rate],
                "Construction Index": [self.construction_index]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)


# Example usage function
def get_house_price_prediction(country, year, rent_index, affordability_ratio, 
                              mortgage_rate, inflation_rate, gdp_growth, 
                              population_growth, urbanization_rate, construction_index):
    """
    Helper function to get house price predictions easily
    
    Returns:
        float: Predicted house price index
    """
    try:
        # Create CustomData object
        data = CustomData(
            country=country,
            year=year,
            rent_index=rent_index,
            affordability_ratio=affordability_ratio,
            mortgage_rate=mortgage_rate,
            inflation_rate=inflation_rate,
            gdp_growth=gdp_growth,
            population_growth=population_growth,
            urbanization_rate=urbanization_rate,
            construction_index=construction_index
        )
        
        # Convert to DataFrame
        df = data.get_data_as_data_frame()
        
        # Initialize prediction pipeline
        predict_pipeline = PredictPipeline()
        
        # Get prediction
        results = predict_pipeline.predict(df)
        
        return results[0]
    
    except Exception as e:
        raise CustomException(e, sys)
