# =======================
# File: download_static_data.py
# =======================
import blpapi
import pandas as pd
from blpapi import SessionOptions, Session

OUTPUT_PATH = "data/nasdaq100_static.csv"

# Fields defined in your project plan
STATIC_FIELDS = [
    "GICS_SECTOR_NAME",  # Sector
    "INDUSTRY_SECTOR",   # Broader Industry
    "CUR_MKT_CAP",       # Market Cap
    "EQY_BETA"           # Beta (Risk metric)
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

# =============================
# STEP 1 — Get Universe (NASDAQ-100)
# =============================
# We fetch the members again to ensure this file matches your price data
print("Fetching NASDAQ-100 constituents...")
ref_request = service.createRequest("ReferenceDataRequest")
ref_request.getElement("securities").appendValue("NDX Index")
ref_request.getElement("fields").appendValue("INDX_MEMBERS")
session.sendRequest(ref_request)

tickers = []
while True:
    event = session.nextEvent(500)
    for msg in event:
        if msg.messageType() == "ReferenceDataResponse":
            data = msg.getElement("securityData").getValue(0).getElement("fieldData")
            if data.hasElement("INDX_MEMBERS"):
                members = data.getElement("INDX_MEMBERS")
                for i in range(members.numValues()):
                    tickers.append(members.getValue(i).getElementAsString("Member Ticker"))
    if event.eventType() == blpapi.Event.RESPONSE:
        break

print(f"Found {len(tickers)} securities.")

# =============================
# STEP 2 — Retrieve Static Data
# =============================
print(f"Requesting static fields: {STATIC_FIELDS}...")
static_request = service.createRequest("ReferenceDataRequest")

for t in tickers:
    static_request.getElement("securities").appendValue(t)

for f in STATIC_FIELDS:
    static_request.getElement("fields").appendValue(f)

session.sendRequest(static_request)

records = []
while True:
    event = session.nextEvent(500)
    for msg in event:
        if msg.messageType() == "ReferenceDataResponse":
            sec_data_array = msg.getElement("securityData")
            for i in range(sec_data_array.numValues()):
                sec_item = sec_data_array.getValue(i)
                security = sec_item.getElementAsString("security")
                field_data = sec_item.getElement("fieldData")
                
                row = {"security": security}
                for field in STATIC_FIELDS:
                    if field_data.hasElement(field):
                        row[field] = field_data.getElementAsString(field) if field != "CUR_MKT_CAP" and field != "EQY_BETA" else field_data.getElementAsFloat(field)
                    else:
                        row[field] = None
                records.append(row)

    if event.eventType() == blpapi.Event.RESPONSE:
        break

# =============================
# STEP 3 — Save to CSV
# =============================
df = pd.DataFrame(records)
df.to_csv(OUTPUT_PATH, index=False)
print("Done! Static data saved to:", OUTPUT_PATH)