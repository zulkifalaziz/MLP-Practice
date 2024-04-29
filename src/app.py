from flask import Flask, request, render_template
import numpy as np
import pandas
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            request.form.get('gender'),
            request.form.get('race_ethnicity'),
            request.form.get('parental_level_of_education'),
            request.form.get('lunch'),
            request.form.get('test_preparation_course'),
            request.form.get('reading_score'),
            request.form.get('writing_score'),
        )

        pred_df = data.get_data_as_dataframe()
        pred = PredictPipeline()
        score = pred.predict(pred_df)

        return render_template('home.html', results=score[0])


if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)