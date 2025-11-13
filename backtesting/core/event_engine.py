from .data_handler import DataHandler


def _extract_prices(state):
    prices = {}
    for column, value in state.items():
        lower = column.lower()
        if "close" in lower or "price" in lower:
            asset = column.split("_")[-1]
            prices[asset] = value
    return prices


def run_backtest(data, strategy, execution, portfolio, analyzer):
    handler = DataHandler(data)
    while True:
        timestamp, state = handler.next()
        if state is None:
            break

        current_prices = _extract_prices(state)
        weights = strategy.generate_weights(state)
        fills = execution.execute(weights, current_prices, timestamp)
        portfolio.update(fills, current_prices, timestamp)

    results = analyzer.analyze(portfolio.history)
    return results, analyzer.results

