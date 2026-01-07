# This ticket expands your model's "training school" from 100 students (NASDAQ) 
# to 500 students (S&P 500).

# 2. Why do this? (The "Global Model" Advantage) Your project uses DeepAR, which is a global 
# forecasting model. Unlike traditional models (like ARIMA) that learn one model per stock, 
# DeepAR learns one universal pattern from thousands of stocks at once.

# More Data = Smarter Model: By feeding it 500 extra stocks (from the S&P 500), the model 
# sees more "examples" of how prices move during crashes, booms, and earnings reports.

# Better Generalization: The NASDAQ-100 is very tech-heavy. The S&P 500 includes 
# Industrials, Energy, and Utilities. Learning these patterns prevents your 
# model from being "overfit" to just tech stocks.

# 3. What data is it pulling?

# Who: It asks Bloomberg for the current members of the SPX Index (S&P 500).

# What: It pulls the same Daily OHLCV (Open, High, Low, Close, Volume) from 2018-01-01 to 2025-01-01.



# =======================
# File: download_sp500_data.py
# =======================
import blpapi
import pandas as pd
from blpapi import SessionOptions, Session

# ------- CONFIGURATION -------
START_DATE = "20180101"
END_DATE   = "20250101"
OUTPUT_PATH = "data/sp500_prices.csv"

# We request the same fields as the NASDAQ dataset
FIELDS = [
    "PX_OPEN",
    "PX_HIGH",
    "PX_LOW",
    "PX_LAST",
    "PX_VOLUME"
]

# ------- BLOOMBERG CONNECTION -------
options = SessionOptions()
options.setServerHost("localhost")
options.setServerPort(8194)

session = Session(options)
if not session.start():
    raise Exception("Failed to start Bloomberg session")

if not session.openService("//blp/refdata"):
    raise Exception("Failed to open //blp/refdata service")

service = session.getService("//blp/refdata")

# ==========================================
# STEP 1 — Retrieve S&P 500 Constituents
# ==========================================
print("Requesting S&P 500 (SPX Index) member list...")

ref_request = service.createRequest("ReferenceDataRequest")
ref_request.getElement("securities").appendValue("SPX Index")
ref_request.getElement("fields").appendValue("INDX_MEMBERS")

session.sendRequest(ref_request)

sp500_members = []

while True:
    event = session.nextEvent(500)
    for msg in event:
        if msg.messageType() == "ReferenceDataResponse":
            sec_data = msg.getElement("securityData").getValue(0)
            field_data = sec_data.getElement("fieldData")

            if field_data.hasElement("INDX_MEMBERS"):
                members = field_data.getElement("INDX_MEMBERS")

                for i in range(members.numValues()):
                    entry = members.getValue(i)
                    # The field name can sometimes vary, so we check possibilities
                    if entry.hasElement("Member Ticker"):
                        ticker = entry.getElementAsString("Member Ticker")
                        # Append " US Equity" if not present (Bloomberg convention)
                        if " Equity" not in ticker:
                            ticker += " US Equity"
                        sp500_members.append(ticker)

    if event.eventType() == blpapi.Event.RESPONSE:
        break

print(f"Found {len(sp500_members)} S&P 500 constituents.")

# ==========================================
# STEP 2 — Retrieve Historical Data (OHLCV)
# ==========================================
print(f"Requesting historical price data ({START_DATE} to {END_DATE})...")

hist_request = service.createRequest("HistoricalDataRequest")

# Add all 500+ tickers to the request
for ticker in sp500_members:
    hist_request.getElement("securities").appendValue(ticker)

for field in FIELDS:
    hist_request.getElement("fields").appendValue(field)

hist_request.set("startDate", START_DATE)
hist_request.set("endDate", END_DATE)
hist_request.set("periodicitySelection", "DAILY")

print("Sending large history request (this may take a moment)...")
session.sendRequest(hist_request)

records = []

# Process the stream of response events
while True:
    event = session.nextEvent(500)
    for msg in event:
        if msg.messageType() == "HistoricalDataResponse":
            security = msg.getElement("securityData").getElementAsString("security")
            field_data = msg.getElement("securityData").getElement("fieldData")

            for i in range(field_data.numValues()):
                fd = field_data.getValueAsElement(i)
                row = {"security": security, "date": fd.getElementAsDatetime("date")}

                for field in FIELDS:
                    if fd.hasElement(field):
                        row[field] = fd.getElementAsFloat(field)
                    else:
                        row[field] = None

                records.append(row)

    if event.eventType() == blpapi.Event.RESPONSE:
        break

# ==========================================
# STEP 3 — Save to CSV
# ==========================================
df = pd.DataFrame(records)
df.to_csv(OUTPUT_PATH, index=False)

print(f"🎉 Done! Saved {len(df)} rows to: {OUTPUT_PATH}")