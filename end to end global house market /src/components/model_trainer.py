import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_model(self, X_train, y_train, X_test, y_test, models, params):
        try:
            report = {}
            
            for model_name, model in models.items():
                # Get parameters for the current model
                model_params = params.get(model_name, {})
                
                # If parameters exist, perform hyperparameter tuning
                if model_params:
                    logging.info(f"Performing hyperparameter tuning for {model_name}")
                    rs = RandomizedSearchCV(
                        model, 
                        model_params, 
                        n_iter=10, 
                        cv=5, 
                        verbose=1, 
                        random_state=42, 
                        n_jobs=-1
                    )
                    rs.fit(X_train, y_train)
                    
                    # Get the best model
                    model = rs.best_estimator_
                    logging.info(f"Best parameters for {model_name}: {rs.best_params_}")
                else:
                    # If no parameters, just fit the model
                    model.fit(X_train, y_train)
                
                # Make predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # Calculate metrics
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                
                # Store results
                report[model_name] = {
                    'model': model,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'test_mae': test_mae,
                    'test_rmse': test_rmse
                }
                
                logging.info(f"{model_name} - Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
            
            return report
            
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Define regression models
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "ElasticNet": ElasticNet(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "K-Neighbors": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False)
            }

            # Define hyperparameters for tuning
            params = {
                "Ridge": {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
                },
                "Lasso": {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
                },
                "ElasticNet": {
                    'alpha': [0.001, 0.01, 0.1, 1.0],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
                },
                "Decision Tree": {
                    'max_depth': [None, 5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "Gradient Boosting": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                "AdaBoost": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.5, 1.0]
                },
                "K-Neighbors": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                "XGBoost": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'colsample_bytree': [0.7, 0.8, 0.9]
                },
                "CatBoost": {
                    'iterations': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'depth': [4, 6, 8]
                }
            }

            # Evaluate models
            model_report = self.evaluate_model(
                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test, 
                models=models, 
                params=params
            )

            # Get the best model based on test R2 score
            best_model_score = -float('inf')
            best_model_name = None
            best_model = None

            for model_name, metrics in model_report.items():
                if metrics['test_r2'] > best_model_score:
                    best_model_score = metrics['test_r2']
                    best_model_name = model_name
                    best_model = metrics['model']

            if best_model_score < 0.6:
                logging.warning(f"Best model R2 score ({best_model_score:.4f}) is below threshold. Consider improving the model.")

            logging.info(f"Best model: {best_model_name} with R2 score: {best_model_score:.4f}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Final prediction and metrics
            y_pred = best_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            logging.info(f"Final model metrics - R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
            
            return {
                'r2_score': r2,
                'mae': mae,
                'rmse': rmse,
                'best_model_name': best_model_name
            }

        except Exception as e:
            raise CustomException(e, sys)
