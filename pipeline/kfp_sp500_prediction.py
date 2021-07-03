import os
import kfp
from kfp.gcp import use_gcp_secret
import kfp.dsl as dsl
import kfp.compiler as compiler
from kfp.components import load_component_from_file

get_data_op = load_component_from_file(
    './containers/get-data/component.yaml')
extract_features_op = load_component_from_file(
    './containers/extract-features/component.yaml')


def get_data(ticker, raw_data_path):
    return get_data_op(ticker, raw_data_path)

def extract_features(raw_data_path, feature_data_path, year_from):
    return extract_features_op(raw_data_path, feature_data_path, year_from)


@dsl.pipeline(
    name='S&P 500 Stock Prediction',
    description='S&P 500 Stock Prediction'
)
def sp500_prediction_pipeline(
        ticker:str='^GSPC', year_from:int=1990, 
        raw_data_path:str='gs://mlops-stock-prediction/raw/sp500.parquet',
        feature_data_path:str='gs://mlops-stock-prediction/feature_store/sp500_features.parquet'):

    get_data_task = get_data(
        ticker=ticker,
        raw_data_path=raw_data_path).set_display_name('Get Data')

    extract_features_task = extract_features(
        raw_data_path=raw_data_path,
        feature_data_path=feature_data_path,
        year_from=year_from).after(get_data_task).set_display_name('Extract Features')


if __name__ == '__main__':
    compiler.Compiler().compile(sp500_prediction_pipeline, __file__ + '.yaml')
