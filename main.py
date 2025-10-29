import pandas as pd
from backtesting.core.data_handler import DataHandler
from backtesting.core.event_engine import run_backtest
from backtesting.core.execution import ExecutionSimulator
from backtesting.core.performance import PerformanceAnalyzer
from backtesting.core.portfolio import PortfolioManager
from backtesting.core.strategy import Strategy

def main():
    print("Setting up and running a sample backtest...")

    data = {
        "AAPL_price": [150, 152, 151, 155, 153],
        "TSLA_price": [800, 810, 790, 820, 815],
        "AAPL_sent": [0.1, 0.2, -0.1, 0.3, 0.0],
        "TSLA_sent": [0.05, -0.05, 0.15, 0.1, -0.02],
    }
    index = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"])
    df = pd.DataFrame(data, index=index)

    strategy = Strategy()  
    execution = ExecutionSimulator()
    portfolio = PortfolioManager(initial_cash=1_000_000)
    analyzer = PerformanceAnalyzer()

    results, _ = run_backtest(df, strategy, execution, portfolio, analyzer)

    print("\anBacktest Results:")
    print(results)

if __name__ == "__main__":
    main()
