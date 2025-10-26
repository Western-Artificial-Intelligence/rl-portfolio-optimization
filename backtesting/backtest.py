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

    def execute_orders(self, target_weights, prices, timestamp):

