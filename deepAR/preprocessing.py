import yfinance as yf

def fetch_daily_stock_data_yf(symbol, start_date, end_date):
    print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
    data = yf.download(symbol, start=start_date, end=end_date)
    
    if data.empty:
        print(f"Warning: No data found for {symbol} between {start_date} and {end_date}")
        return None
    
    data.reset_index(inplace=True)
    
    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    data.rename(columns={'Date': 'date'}, inplace=True)
    
    data['date'] = data['date'].dt.strftime('%Y-%m-%d')
    
    csv_filename = f"{symbol}_data.csv"
    data.to_csv(csv_filename, index=False)
    
    print(f"Successfully saved {len(data)} records to {csv_filename}")
    print(f"Date range: {data['date'].iloc[0]} to {data['date'].iloc[-1]}")
    
    return csv_filename


if __name__ == "__main__":
    symbol = "SQQQ"
    start_date = "2005-01-01"
    end_date = "2023-12-31"
    
    try:
        filename = fetch_daily_stock_data_yf(symbol, start_date, end_date)
        if filename:
            print(f"\nData saved to: {filename}")
    except Exception as e:
        print(f"Error: {e}")
