import sys
from src.exception import CustomError
import pandas as pd
from src.utils import load_object
pd.set_option('display.max_columns', None)

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, data_point):
        try:
            model_path = 'C:/Users/zulki/Desktop/MLP-Practice/src/artifacts/model.pkl'
            preprocessor_path = 'C:/Users/zulki/Desktop/MLP-Practice/src/artifacts/preprocessor.pkl'
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)


            data_point = preprocessor.transform(data_point)
            pred = model.predict(data_point)

            return pred
        except Exception as e:
            raise CustomError(e, sys)
        pass

class CustomData:
    def __init__(self,
                 gender,
                 race_ethnicity,
                 parental_level_of_education,
                 lunch,
                 test_preparation_course,
                 reading_score,
                 writing_score):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        pass

    def get_data_as_dataframe(self):
        try:
            data = {
                'writing_score': [self.writing_score],
                'reading_score': [self.reading_score],
                'gender': [self.gender],
                'race_ethnicity': [self.race_ethnicity],
                'parental_level_of_education': [self.parental_level_of_education],
                'lunch': [self.lunch],
                'test_preparation_course': [self.test_preparation_course],

            }
            df = pd.DataFrame(data)
            return df
        except Exception as e:
            raise CustomError(e, sys)
        pass
