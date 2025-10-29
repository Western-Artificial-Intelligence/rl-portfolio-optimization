import pandas as pd 
import numpy as np

class PerformanceAnalyzer:
    def __init__(self):
        self.results = pd.DataFrame();

    def analyze(self, portfolio_history):
        df = pd.DataFrame(portfolio_history).set_index("time");
        df["returns"] = df["value"].pct_change();
        self.results = df;
        return df;

