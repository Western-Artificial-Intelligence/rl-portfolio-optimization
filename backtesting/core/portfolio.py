class PortfolioManager:
    def __init__(self, initial_cash = 1_000_000):
        self.cash = initial_cash;
        self.positions = {};
        self.history = [];

    def update(self, fills, current_prices, timestamp):
        total_value = 0;
        for asset, fill in fills.items():
            shares = fill["value"] / fill["price"];
            self.positions[asset] = shares;
            total_value += shares * current_prices[f"{asset}_price"];
        self.cash = 0;
        self.history.append({"time": timestamp, "value": total_value});
        return total_value;
