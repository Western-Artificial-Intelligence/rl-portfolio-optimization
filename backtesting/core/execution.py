class ExecutionSimulator:
    def __init__(self, slippage=0.001, transaction_fee=0.0005):
        self.slippage = slippage;
        self.transaction_fee = transaction_fee;

    def execute(self, target_weights, current_prices, portfolio_value):
        fills = {};
        for asset, w in target_weights.items():
            fill_price = current_prices[f"{asset}_price"] * (1 + self.slippage);
            fills[asset] = {
                "price": fill_price,
                "allocation": w,
                "value": w * portfolio_value * (1 - self.transaction_fee)
            }
        return fills;
