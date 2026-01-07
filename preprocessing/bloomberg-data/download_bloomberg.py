# =======================
# File: download_bloomberg_data.py
# =======================
import blpapi
import pandas as pd
from datetime import datetime
from blpapi import SessionOptions, Session

# ------- CONFIGURATION -------
START_DATE = "20180101"
END_DATE = "20250101"
OUTPUT_PATH = "data/bloomberg_prices.csv"

SECURITIES = [
    "META US Equity",
    "MSFT US Equity",
    "AAPL US Equity",
    "NVDA US Equity",
    "AMZN US Equity",
    "TSLA US Equity",
    "PSQ US Equity",    # inverse ETF
    "SPX Index",        # S&P 500 benchmark
    "NDX Index",        # Nasdaq 100 (MegaTech proxy)
    "VIX Index"         # volatility index
]

FIELDS = [
    "PX_OPEN",
    "PX_HIGH",
    "PX_LOW",
    "PX_LAST",
    "PX_VOLUME"
]

# ------- BLOOMBERG SESSION SETUP -------
options = SessionOptions()
options.setServerHost("localhost")
options.setServerPort(8194)

session = Session(options)
if not session.start():
    raise Exception("Failed to start Bloomberg session")

if not session.openService("//blp/refdata"):
    raise Exception("Failed to open //blp/refdata service")

service = session.getService("//blp/refdata")
request = service.createRequest("HistoricalDataRequest")

for sec in SECURITIES:
    request.getElement("securities").appendValue(sec)

for field in FIELDS:
    request.getElement("fields").appendValue(field)

request.set("startDate", START_DATE)
request.set("endDate", END_DATE)
request.set("periodicitySelection", "DAILY")

print("Sending Bloomberg data request...")
session.sendRequest(request)

# ------- PARSING RESPONSE -------
records = []

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

# Convert to CSV
df = pd.DataFrame(records)
df.to_csv(OUTPUT_PATH, index=False)

print("Done! Saved to:", OUTPUT_PATH)