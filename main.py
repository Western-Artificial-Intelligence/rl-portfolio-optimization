import pandas as pd
from backtesting.core.PortfolioEnv import PortfolioEnv

def main():
    print("Setting up and running a sample environment...")

    csv_file_path = "data.csv"
    df_csv = pd.read_csv(csv_file_path, index_col="date", parse_dates=True)

    # Transform df_csv to match PortfolioEnv expectations
    df = pd.DataFrame(index=df_csv.index)
    df["SQQQ"] = df_csv["Close_SQQQ"]
    # Generate some sample sentiment data for SQQQ
    sentiment_values = [0.1 if i % 2 == 0 else -0.1 for i in range(len(df_csv))]
    df["sentiment_SQQQ"] = sentiment_values

    assets = ["SQQQ"]
    env = PortfolioEnv(df, assets)

    obs, _ = env.reset()
    for _ in range(100):
        action = env.action_space.sample()  # Replace with your agent's action
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        if done:
            break

    print("\nEnvironment run complete.")

if __name__ == "__main__":
    main()

