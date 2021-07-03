import yfinance as yf
import argparse
from pathlib import Path


'''download data from Yahoo Finance'''

def download_raw_data(ticker, raw_data_path):
    sp500_df = yf.download(ticker, progress=False)
    print('Downloaded data for ' + ticker)
    print(sp500_df.head())
    print('trying to write the data to ' + raw_data_path)
    sp500_df.to_parquet(raw_data_path, compression='GZIP')
    print('Done!')

def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--ticker', help="ticker for which to download data")
    parser.add_argument('-d', '--raw_data_path', help="output data path")
    args = parser.parse_args()
    return vars(args)

if __name__ == '__main__':
    download_raw_data(**_cli())