class PortfolioManager:
    def __init__(self, initial_cash = 1_000_000):
        self.cash = initial_cash;
        self.positions = {};
        self.history = [];

    def update(self, fills, current_prices, timestamp):
        for asset, fill in fills.items():
            target_value = fill["value"]
            fill_price = fill["price"]

            current_shares = self.positions.get(asset, 0)
            price_key = f"{asset}_price"
            mark_price = current_prices.get(price_key, current_prices.get(asset))
            if mark_price is None:
                raise KeyError(f"Missing price for {asset}")

            current_asset_value = current_shares * mark_price

            value_to_trade = target_value - current_asset_value

            if value_to_trade > 0: 
                shares_to_buy = value_to_trade / fill_price
                cost = shares_to_buy * fill_price
                self.cash -= cost
                self.positions[asset] = current_shares + shares_to_buy
            elif value_to_trade < 0: 
                shares_to_sell = abs(value_to_trade) / fill_price
                revenue = shares_to_sell * fill_price
                self.cash += revenue
                self.positions[asset] = current_shares - shares_to_sell

        new_total_portfolio_value = self.cash
        for asset, shares in self.positions.items():
            price_key = f"{asset}_price"
            mark_price = current_prices.get(price_key, current_prices.get(asset))
            if mark_price is not None:
                new_total_portfolio_value += shares * mark_price

        self.history.append({"time": timestamp, "value": new_total_portfolio_value});

        return new_total_portfolio_value;
