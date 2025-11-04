import yfinance as yf
import pandas as pd

def fetch_daily_stock_data_yf(start_date, end_date, tickers = []):
    if not tickers:
        print("Warning: No tickers provided")
        return None
    
    all_data = []
    
    for symbol in tickers:
        print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
        data = yf.download(symbol, start=start_date, end=end_date)
        
        if data.empty:
            print(f"Warning: No data found for {symbol} between {start_date} and {end_date}")
            continue
        
        data.reset_index(inplace=True)
        
        normalized_data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        normalized_data.rename(columns={
            'Date': 'Datetime'
        }, inplace=True)
        
        normalized_data['Symbol'] = symbol
        
        normalized_data = normalized_data[['Datetime', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        all_data.append(normalized_data)
        print(f"Successfully fetched {len(normalized_data)} records for {symbol}")
        print(f"Date range: {normalized_data['Datetime'].iloc[0]} to {normalized_data['Datetime'].iloc[-1]}")
    
    if not all_data:
        print("No data was successfully fetched for any ticker")
        return None
    
    merged_data = pd.concat(all_data, ignore_index=True)
    
    merged_data['Datetime'] = pd.to_datetime(merged_data['Datetime'])
    merged_data.sort_values(['Datetime', 'Symbol'], inplace=True)
    merged_data['Datetime'] = merged_data['Datetime'].dt.strftime('%Y-%m-%d')
    
    if len(tickers) == 1:
        csv_filename = f"{tickers[0]}_data.csv"
    else:
        csv_filename = f"{'_'.join(tickers)}_aggregated_data.csv"
    
    merged_data.to_csv(csv_filename, index=False)
    
    print(f"\nSuccessfully saved aggregated data with {len(merged_data)} records to {csv_filename}")
    print(f"Date range: {merged_data['Datetime'].min()} to {merged_data['Datetime'].max()}")
    print(f"Symbols: {sorted(merged_data['Symbol'].unique())}")
    print(f"Columns: {list(merged_data.columns)}")
    
    return csv_filename


if __name__ == "__main__":
    tickers = ["SQQQ", "AAPL", "PLTR"]
    start_date = "2005-01-01"
    end_date = "2023-12-31"
    
    try:
        filename = fetch_daily_stock_data_yf(start_date, end_date, tickers)
        if filename:
            print(f"\nData saved to: {filename}")
    except Exception as e:
        print(f"Error: {e}")
