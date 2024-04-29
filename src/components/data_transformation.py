import sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import os
from src.exception import CustomError
from src.logger import logging
import numpy as np
from src.utils import save_object


@dataclass
class DataTransformConfig:
    preprocessor_obj_file_path = os.path.join('../artifacts', 'preprocessor.pkl')
    pass


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformConfig
        pass

    def get_data_transformation_obj(self):
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            num_pipeline = Pipeline(
                steps=[
                    ('impute',SimpleImputer(strategy='median')),
                    ('scalar',StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('impute', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scalar', StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            CustomError(e, sys)
        pass

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformation_obj()

            logging.info('Transformation object successfully created!')

            target_column_name = 'math_score'
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]


            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomError(e, sys)
        pass


if __name__ == '__main__':
    from src.components.data_ingestion import DataInjestion

    obj1 = DataInjestion()
    train_path, test_path = obj1.injestion()

    obj2 = DataTransformation()
    train_arr, test_arr, path = obj2.initiate_data_transformation(train_path, test_path)
    print(train_arr[0])