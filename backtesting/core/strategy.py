class Strategy:
    def __init__(self, model=None):
        self.model = model;

    def generate_weights(self, state):
        if self.model:
            return self.model.predict(state);
        
        weights = {
            "AAPL": max(0, state["AAPL_sent"]),
            "TSLA": max(0, state["TSLA_sent"])
        };

        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}
