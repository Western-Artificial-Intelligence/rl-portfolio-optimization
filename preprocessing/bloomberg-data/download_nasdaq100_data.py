# =======================
# File: download_nasdaq100_data.py
# =======================
import blpapi
import pandas as pd
from blpapi import SessionOptions, Session

START_DATE = "20180101"
END_DATE   = "20250101"
OUTPUT_PATH = "data/nasdaq100_prices.csv"

FIELDS = [
    "PX_OPEN",
    "PX_HIGH",
    "PX_LOW",
    "PX_LAST",
    "PX_VOLUME"
]

# Bloomberg Session Setup
options = SessionOptions()
options.setServerHost("localhost")
options.setServerPort(8194)

session = Session(options)
if not session.start():
    raise Exception("Failed to start Bloomberg session")

if not session.openService("//blp/refdata"):
    raise Exception("Failed to open //blp/refdata service")

service = session.getService("//blp/refdata")

# =============================
# STEP 1 — Retrieve NASDAQ-100 Members
# =============================
ref_request = service.createRequest("ReferenceDataRequest")
ref_request.getElement("securities").appendValue("NDX Index")
ref_request.getElement("fields").appendValue("INDX_MEMBERS")

print("Requesting NASDAQ-100 member list...")
session.sendRequest(ref_request)

nasdaq_members = []

while True:
    event = session.nextEvent()
    for msg in event:
        if msg.messageType() == "ReferenceDataResponse":
            sec_data = msg.getElement("securityData").getValue(0)
            field_data = sec_data.getElement("fieldData")

            if field_data.hasElement("INDX_MEMBERS"):
                members = field_data.getElement("INDX_MEMBERS")

                for i in range(members.numValues()):
                    entry = members.getValue(i)

                    # Try different possible key names
                    if entry.hasElement("Member Ticker"):
                        ticker = entry.getElementAsString("Member Ticker")
                    elif entry.hasElement("security"):
                        ticker = entry.getElementAsString("security")
                    elif entry.hasElement("Security"):
                        ticker = entry.getElementAsString("Security")
                    else:
                        print("⚠️ Unknown structure for entry:", entry)
                        continue

                    nasdaq_members.append(ticker)

    if event.eventType() == blpapi.Event.RESPONSE:
        break

print(f"Found {len(nasdaq_members)} NASDAQ-100 constituents.")
print(nasdaq_members)

# =============================
# STEP 2 — Retrieve Historical OHLCV Data
# =============================
hist_request = service.createRequest("HistoricalDataRequest")

for ticker in nasdaq_members:
    hist_request.getElement("securities").appendValue(ticker)

for field in FIELDS:
    hist_request.getElement("fields").appendValue(field)

hist_request.set("startDate", START_DATE)
hist_request.set("endDate", END_DATE)
hist_request.set("periodicitySelection", "DAILY")

print("\nRequesting price history from Bloomberg...")
session.sendRequest(hist_request)

records = []

while True:
    event = session.nextEvent(500)
    for msg in event:
        if msg.messageType() == "HistoricalDataResponse":
            security = msg.getElement("securityData").getElementAsString("security")
            fdata = msg.getElement("securityData").getElement("fieldData")

            for i in range(fdata.numValues()):
                fd = fdata.getValueAsElement(i)
                row = {"security": security, "date": fd.getElementAsDatetime("date")}

                for field in FIELDS:
                    row[field] = fd.getElementAsFloat(field) if fd.hasElement(field) else None

                records.append(row)

    if event.eventType() == blpapi.Event.RESPONSE:
        break

df = pd.DataFrame(records)
df.to_csv(OUTPUT_PATH, index=False)

print("\n🎉 Done! Saved NASDAQ-100 dataset to:", OUTPUT_PATH)