def run_backtest(data, strategy, execution, portfolio, analyzer):
    handler = DataHandler(data);
    while True:
        timestamp, state = handler.next();
        if state is None:
            break;

        current_prices = {k: v for k, v in state.items() if "price" in k};
        weights = strategy.generate_weights(state);
        fills = execution.execute(weights, current_prices, portfolio.history[-1]["value"] if portfolio.history else portfolio.cash);
        portfolio.update(fills, current_prices, timestamp);

    results = analyzer.analyze(portfolio.history);
    return results, analyzer.results;

