class Strategy:
    def __init__(self, model=None):
        self.model = model;

    def generate_weights(self, state):
        if self.model:
            return self.model.predict(state);
        
        sentiment = state.get("sentiment_SQQQ", state.get("SQQQ_sent", 0))
        weights = {
            "SQQQ": max(0, sentiment)
        };

        total = sum(weights.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in weights.items()}
