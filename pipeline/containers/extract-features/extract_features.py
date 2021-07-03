import argparse
import pandas as pd
from datetime import datetime

'''calculate features for our machine learning model'''

def feature_processing(raw_data_path, feature_data_path, year_from):
    
    # read dataframe
    sp500_df = pd.read_parquet(raw_data_path)
    
    # create empty df to store feature
    sp500_feautres_df = pd.DataFrame()
    
    average_days_window_closing_price = [5, 30, 120, 365]
    # average price for window of different days
    for window in average_days_window_closing_price:
        sp500_feautres_df['Close__rolling_mean__'+str(window)+'_days'] = sp500_df['Close'].rolling(window).mean().shift(periods=1)
        sp500_feautres_df['Close__rolling_std__'+str(window)+'_days'] = sp500_df['Close'].rolling(window).std().shift(periods=1)
        sp500_feautres_df['Close__rolling_max__'+str(window)+'_days'] = sp500_df['Close'].rolling(window).max().shift(periods=1)
        sp500_feautres_df['Close__rolling_min__'+str(window)+'_days'] = sp500_df['Close'].rolling(window).min().shift(periods=1)
        sp500_feautres_df['Close__rolling_range__'+str(window)+'_days'] = sp500_feautres_df['Close__rolling_max__'+str(window)+'_days'] - sp500_feautres_df['Close__rolling_min__'+str(window)+'_days']
    
    average_days_window_volume = [5, 10, 15]
    # average price for window of different days
    for window in average_days_window_volume:
        sp500_feautres_df['Volume__rolling_max__'+str(window)+'_days'] = sp500_df['Close'].rolling(window).max().shift(periods=1)
        sp500_feautres_df['Volume__rolling_sum__'+str(window)+'_days'] = sp500_df['Close'].rolling(window).sum().shift(periods=1)
        
    # get day of the week
    sp500_df['day_of_week'] = sp500_df.index.dayofweek
    # get quarter
    sp500_df['quarter'] = sp500_df.index.quarter
    
    sp500_feautres_df = pd.concat([sp500_feautres_df, pd.get_dummies(sp500_df['day_of_week'], prefix='day_of_week')], 1)
    sp500_feautres_df = pd.concat([sp500_feautres_df, pd.get_dummies(sp500_df['day_of_week'], prefix='quarter')], 1)
    
    # let's not confuse our model from data from way back
    sp500_feautres_df = sp500_feautres_df[sp500_feautres_df.index > datetime(year=1990, month=12, day=31)]
    # get label for feature dataset
    sp500_timeboxed_feautres_df = pd.merge(sp500_df['Close'], sp500_feautres_df, left_index=True, right_index=True)
    # write out to parquet
    sp500_timeboxed_feautres_df.to_parquet(feature_data_path, compression='GZIP')
    features_numbers = len(sp500_timeboxed_feautres_df.columns) - 1
    total_days = len(sp500_timeboxed_feautres_df)
    print('Writing %s features for %s days' % (features_numbers, total_days))
    print('Done!')
    
    return feature_data_path

def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--raw_data_path', help="data path for raw data")
    parser.add_argument('-o', '--feature_data_path', help="path for saving processed data")
    parser.add_argument('-y', '--year_from', type=int, help="data beyond this year will be dropped")
    args = parser.parse_args()
    return vars(args)

if __name__ == '__main__':
    feature_processing(**_cli())