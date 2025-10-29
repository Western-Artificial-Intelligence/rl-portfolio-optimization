import pandas as pd
from backtesting.core.data_handler import DataHandler
from backtesting.core.event_engine import run_backtest
from backtesting.core.execution import ExecutionSimulator
from backtesting.core.performance import PerformanceAnalyzer
from backtesting.core.portfolio import PortfolioManager
from backtesting.core.strategy import Strategy

def main():
    print("Setting up and running a sample backtest...")

    csv_file_path = "data.csv" # Assuming data.csv is in the same directory
    df_csv = pd.read_csv(csv_file_path, index_col="date", parse_dates=True)

    # Transform df_csv to match run_backtest expectations
    df = pd.DataFrame(index=df_csv.index)
    df["SQQQ_price"] = df_csv["Close_SQQQ"]
    # Generate some sample sentiment data for SQQQ_sent
    # For demonstration, let's alternate positive and negative sentiment
    sentiment_values = [0.1 if i % 2 == 0 else -0.1 for i in range(len(df_csv))]
    df["SQQQ_sent"] = sentiment_values


    strategy = Strategy()  
    execution = ExecutionSimulator()
    portfolio = PortfolioManager(initial_cash=1_000_000)
    analyzer = PerformanceAnalyzer()

    results, _ = run_backtest(df, strategy, execution, portfolio, analyzer)

    print("\anBacktest Results:")
    print(results)

if __name__ == "__main__":
    main()
