import numpy as np 
import pandas as pd
from copy import deepcopy

class ExecutionSimulator:

    def __init__(self, initial_cash=1_000_000, slippage_bps=5):
        self.cash = initial_cash;
        self.holdings = {};
        self.portfolio_value = initial_cash;
        self.slippage_bps = slippage_bps / 10_000;
        self.history = [];

    def _get_portfolio_value(self, prices):
        holdings_value = sum(self.holdings.get(a, 0) * prices[a] for a in prices);
        return self.cash + holdings_value;

    def _apply_slippage(self, price, side):
        if side == "buy":
            return price * (1+ self.slippage_bps);
        elif side == "sell":
            return price *(1-self.slippage_bps);
        return price

    def execute_orders(self, target_weights, prices, timestamp):
        current_value = self._get_portfolio_value(prices);
        current_alloc = {
            a: self.holdings.get(a,0) * prices[a] for a in prices
        }
        target_alloc = {
            a: target_weights.get(a,0) * current_value for a in prices
        }

        orders = {};
        for a in prices:
            diff_value = target_alloc[a] - current_alloc.get(a,0);
            if abs(diff_value) > 1e-8:
                shares_to_trade = diff_value / prices[a];
                side = "buy" if shares_to_trade > 0 else "sell";
                trade_price = self._apply_slippage(prices[a], side);
                orders[a] = (shares_to_trade, trade_price);

        for a, (shares, trade_price) in orders.items():
            cost = shares * trade_price;
            self.holdings[a] = self.holdings.get(a,0) + shares;
            self.cash += cost;

        new_value = self._get_portfolio_value(prices);
        self.portfolio_value = new_value;
        
        self.history.append({
            "timestamp": timestamp,
            "portfolio_value": new_value, 
            "cash": self.cash,
            "holdings": deepcopy(self.holdings),
            "prices": deepcopy(prices),
            "target_weights": deepcopy(target_weights)
        })

    def get_history(self):
        df = pd.DataFrame(self.history);
        df.set_index("timestamp", inplace = True);
        return df;





