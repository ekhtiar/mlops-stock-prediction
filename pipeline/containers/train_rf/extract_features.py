import argparse
import pandas as pd
from datetime import datetime, timedelta
import _pickle as cPickle # save ML model
from google.cloud import storage # save the model to GCS
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from urllib.parse import urlparse

'''train a random forest model with default parameters'''

def train_vanilla_rf(feature_data_path, vanilla_model_path, holdout_days):
    
    # read dataframe
    sp500_timeboxed_feautres_df = pd.read_parquet(feature_data_path)
    
    # this will be our training set
    sp500_train_df = sp500_timeboxed_feautres_df[sp500_timeboxed_feautres_df.index < (datetime.today() - timedelta(days=holdout_days))]
    
    # get x and y
    x_train, y_train = sp500_train_df.drop('Close', axis=1), sp500_train_df['Close']
    # split the data for initial testing
    X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2,random_state=786)
    
    # train the model
    print('Training vanilla Random Forest models')
    print('Shape of X: %s, %s' % (len(x_train), len(x_train.columns)))
    vanilla_rf = RandomForestRegressor()
    vanilla_rf.fit(X_train, Y_train)
    
    # some initial testing
    predictions_vanilla_rf = vanilla_rf.predict(X_test)
    print('mean absolute error without optimization: %s' % mean_absolute_error(Y_test, predictions_vanilla_rf))
    print('mean squared error without optimization is: %s' % mean_squared_error(Y_test, predictions_vanilla_rf)) 
    
    # write out output
    # save the model into temp
    with open('/tmp/model.pickle', 'wb') as f:
        cPickle.dump(vanilla_rf, f, -1)
        
    # get client and write to GCS
    # parse model write path for GS
    parse = urlparse(url=vanilla_model_path, allow_fragments=False)
    if parse.path[0] =='/':
        model_path = parse.path[1:]
        
    client = storage.Client()
    bucket = client.get_bucket(parse.netloc)
    blob = bucket.blob(model_path)
    blob.upload_from_filename('/tmp/model.pickle')
    
    return vanilla_model_path

def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--raw_data_path', help="data path for raw data")
    parser.add_argument('-o', '--feature_data_path', help="path for saving processed data")
    parser.add_argument('-y', '--year_from', type=int, help="data beyond this year will be dropped")
    args = parser.parse_args()
    return vars(args)

if __name__ == '__main__':
    feature_processing(**_cli())