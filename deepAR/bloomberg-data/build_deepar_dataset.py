# =======================
# File: build_deepar_dataset.py
# =======================
import pandas as pd
import numpy as np

INPUT_PATH = "data/bloomberg_prices.csv"
OUTPUT_PATH = "data/deepar_dataset.csv"

df = pd.read_csv(INPUT_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["security", "date"]).reset_index(drop=True)

# -------- FEATURE ENGINEERING --------
# Compute log returns
df["log_return"] = df.groupby("security")["PX_LAST"].apply(
    lambda x: np.log(x / x.shift(1))
)

# Rolling volatility (10-day realized volatility)
df["roll_vol_10"] = df.groupby("security")["log_return"].rolling(10).std().reset_index(level=0, drop=True)

# Rolling momentum returns
df["ret_5d"] = df.groupby("security")["PX_LAST"].pct_change(5)
df["ret_20d"] = df.groupby("security")["PX_LAST"].pct_change(20)

# Volume z-score
df["zvol"] = df.groupby("security")["PX_VOLUME"].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Drop NA rows created by rolling windows
df = df.dropna().reset_index(drop=True)

# -------- SAVE DATASET FOR DEEPAR --------
df.to_csv(OUTPUT_PATH, index=False)
print("DeepAR formatted dataset saved to:", OUTPUT_PATH)