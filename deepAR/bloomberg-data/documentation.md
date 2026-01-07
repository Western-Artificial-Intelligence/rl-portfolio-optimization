# Folder: deepAR/bloomberg-data/

This folder contains the **Extraction** and **Transformation** pipeline for the project. Its goal is to turn raw financial data from the Bloomberg Terminal into a "feature-rich" dataset ready for the DeepAR forecasting model.

---

## 1. Script-by-Script Documentation

### A. `test-bloomberg.py`
* **Purpose:** A sanity check script to verify connectivity.
* **Function:** It attempts to open a session with the local Bloomberg API (`localhost:8194`). It requests a single data point (e.g., AAPL price) to ensure the terminal is logged in and the API service is responding.
* **When to run:** Run this first if you are getting connection errors in other scripts.

### B. `download_nasdaq100_data.py` (Step 1a: Core Universe Extraction)
* **Purpose:** Fetches historical price movements for the entire NASDAQ-100 universe.
* **Key Actions:**
    * **Dynamic Universe:** Connects to Bloomberg (`//blp/refdata`) and requests the current members of the `NDX Index`.
    * **History Request:** Iterates through those ~100 tickers and requests daily OHLCV data (Open, High, Low, Last, Volume) from 2018 to 2025.
* **Output:** Saves raw price history to `data/nasdaq100_prices.csv`.

### C. `download_sp500_data.py` (Step 1b: Extended Universe Extraction)
* **Purpose:** Fetches historical price movements for the S&P 500 (Extended Dataset) to improve model generalization.
* **Key Actions:**
    * **Dynamic Universe:** Connects to Bloomberg (`//blp/refdata`) and requests the current members of the `SPX Index`.
    * **History Request:** Iterates through ~500 tickers and requests daily OHLCV data for the same period (2018–2025).
* **Output:** Saves raw price history to `data/sp500_prices.csv`.

### D. `download_static_data.py` (Step 2: Context Enrichment)
* **Purpose:** Fetches "Static" metadata that doesn't change daily but is crucial for categorical embedding.
* **Key Actions:**
    * **Universe Match:** Fetches the constituents to ensure alignment with the price scripts.
    * **Reference Request:** Pulls specific fields that `HistoricalDataRequest` cannot handle:
        * `GICS_SECTOR_NAME`: For learning sector correlations.
        * `CUR_MKT_CAP`: To distinguish between large and small caps.
        * `EQY_BETA`: To measure systemic risk.
* **Output:** Saves metadata to `data/nasdaq100_static.csv`.

### E. `build_deepar_dataset.py` (Step 3: Feature Engineering)
* **Purpose:** The "Bridge" between raw Bloomberg CSVs and the DeepAR model. It transforms raw prices into stationary features.
* **Key Actions:**
    * **Ingestion:** Reads the raw CSVs produced by Step 1a, 1b, and Step 2.
    * **Transformation:**
        * **Log Returns:** $\ln(P_t / P_{t-1})$ (Stationarity).
        * **Rolling Volatility:** 10-day standard deviation (Risk metric).
        * **Momentum:** 5-day and 20-day percentage changes.
        * **Volume Z-Score:** Normalizes volume to detect anomalies.
    * **Merging:** Joins the static data (Sector/Beta) to each time step.
* **Output:** Saves the final training set to `data/deepar_dataset.csv`.

---

## 2. Interaction & Data Flow

The scripts interact via intermediary CSV files stored in the `data/` directory.

```mermaid
graph TD
    A[Bloomberg Terminal] -->|API Connection| B(download_nasdaq100_data.py)
    A -->|API Connection| C(download_sp500_data.py)
    A -->|API Connection| D(download_static_data.py)
    
    B -->|Outputs| E[data/nasdaq100_prices.csv]
    C -->|Outputs| F[data/sp500_prices.csv]
    D -->|Outputs| G[data/nasdaq100_static.csv]
    
    E --> H(build_deepar_dataset.py)
    F --> H
    G --> H
    
    H -->|Outputs| I[Final: data/deepar_dataset.csv]


    Text Diagram:
    [ Bloomberg Terminal ]
       |
       v
---------------------------      ---------------------------
1. download_nasdaq100.py            2. download_sp500.py
   (Step 1a)                           (Step 1b)
   --> data/nasdaq100_prices.csv       --> data/sp500_prices.csv
---------------------------      ---------------------------
       |                                   |
       v                                   v
---------------------------
3. download_static_data.py  (Step 2)
   --> data/nasdaq100_static.csv
---------------------------
              |
              v
---------------------------
4. build_deepar_dataset.py (Step 3)
   (Consolidates NASDAQ + SP500, Calculates Features)
---------------------------
              |
              v
[ Final Output: data/deepar_dataset.csv ]

How to run teh scripts:
S1: Loginto a Bloomberg Terminal with API
S2: Extract Data
python deepAR/bloomberg-data/download_nasdaq100_data.py
python deepAR/bloomberg-data/download_sp500_data.py
python deepAR/bloomberg-data/download_static_data.py

S3:Transform (Feature Engineering):
python deepAR/bloomberg-data/build_deepar_dataset.py
