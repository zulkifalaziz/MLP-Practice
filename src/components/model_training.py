import os
import sys
from dataclasses import dataclass
from xgboost import XGBRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from src.exception import CustomError
from src.logger import logging
from src.utils import evaluate
from src.components.data_ingestion import DataInjestion
from src.components.data_transformation import DataTransformation
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('../artifacts','model.pkl')
    pass

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        pass

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "K-Neighbor Regressor": KNeighborsRegressor()
            }

            model_report = evaluate(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            best_model_score = max(list(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            model = models[best_model_name]
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            r2 = r2_score(y_test, predictions)

            return r2

        except Exception as e:
            raise CustomError(e, sys)
        pass

if __name__ == '__main__':
    obj_1 = DataInjestion()
    train_path, test_path = obj_1.injestion()

    obj_2 = DataTransformation()
    train_arr, test_arr, _ = obj_2.initiate_data_transformation(train_path, test_path)

    obj_3 = ModelTrainer()
    report = obj_3.initiate_model_trainer(train_arr, test_arr)

    print(report)