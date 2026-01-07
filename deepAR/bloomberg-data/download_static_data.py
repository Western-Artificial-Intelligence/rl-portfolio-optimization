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

def main():
    # ------- BLOOMBERG CONNECTION -------
    options = SessionOptions()
    options.setServerHost("localhost")
    options.setServerPort(8194)

    session = Session(options)
    if not session.start():
        print("❌ Failed to start Bloomberg session")
        return

    if not session.openService("//blp/refdata"):
        print("❌ Failed to open //blp/refdata service")
        return

    service = session.getService("//blp/refdata")

    # =============================
    # STEP 1 — Get Universe (NASDAQ-100)
    # =============================
    print("Fetching NASDAQ-100 constituents...")
    ref_request = service.createRequest("ReferenceDataRequest")
    ref_request.getElement("securities").appendValue("NDX Index")
    ref_request.getElement("fields").appendValue("INDX_MEMBERS")
    session.sendRequest(ref_request)

    tickers = []
    while True:
        event = session.nextEvent(500)
        if event.eventType() == blpapi.Event.RESPONSE:
            for msg in event:
                if msg.messageType() == "ReferenceDataResponse":
                    if msg.hasElement("securityData"):
                        # securityData is an array, we must loop through it
                        sec_data_array = msg.getElement("securityData")
                        for k in range(sec_data_array.numValues()):
                            sec_item = sec_data_array.getValue(k)
                            field_data = sec_item.getElement("fieldData")
                            
                            if field_data.hasElement("INDX_MEMBERS"):
                                members = field_data.getElement("INDX_MEMBERS")
                                for i in range(members.numValues()):
                                    member_data = members.getValue(i)
                                    
                                    # --- FIX: ROBUST FIELD CHECKING ---
                                    t = None
                                    if member_data.hasElement("Member Ticker and Exchange Code"):
                                        t = member_data.getElementAsString("Member Ticker and Exchange Code")
                                    elif member_data.hasElement("Member Ticker"):
                                        t = member_data.getElementAsString("Member Ticker")
                                    
                                    if t:
                                        tickers.append(t)
            break
        elif event.eventType() == blpapi.Event.TIMEOUT:
            continue

    print(f"Found {len(tickers)} securities.")

    if not tickers:
        print("❌ No tickers found. Exiting.")
        return

    # ---------------------------------------------------------
    # STEP 2: Normalize Tickers
    # ---------------------------------------------------------
    # 'AAPL UW' -> 'AAPL US Equity'
    clean_tickers = [f"{t.split()[0]} US Equity" for t in tickers]
    print(f"Sample normalized tickers: {clean_tickers[:3]}")

    # =============================
    # STEP 3 — Retrieve Static Data
    # =============================
    print(f"Requesting static fields: {STATIC_FIELDS}...")
    
    # We can request all 100 tickers at once for ReferenceData (it handles larger batches better than History)
    static_request = service.createRequest("ReferenceDataRequest")

    for t in clean_tickers:
        static_request.getElement("securities").appendValue(t)

    for f in STATIC_FIELDS:
        static_request.getElement("fields").appendValue(f)

    session.sendRequest(static_request)

    records = []
    while True:
        event = session.nextEvent(500)
        
        # Process responses
        if event.eventType() in [blpapi.Event.RESPONSE, blpapi.Event.PARTIAL_RESPONSE]:
            for msg in event:
                if msg.messageType() == "ReferenceDataResponse":
                    if msg.hasElement("securityData"):
                        sec_data_array = msg.getElement("securityData")
                        for i in range(sec_data_array.numValues()):
                            sec_item = sec_data_array.getValue(i)
                            security = sec_item.getElementAsString("security")
                            
                            if sec_item.hasElement("fieldData"):
                                field_data = sec_item.getElement("fieldData")
                                
                                row = {"security": security}
                                for field in STATIC_FIELDS:
                                    if field_data.hasElement(field):
                                        # Handle Float vs String types safely
                                        if field in ["CUR_MKT_CAP", "EQY_BETA"]:
                                             row[field] = field_data.getElementAsFloat(field)
                                        else:
                                             row[field] = field_data.getElementAsString(field)
                                    else:
                                        row[field] = None
                                records.append(row)
            
            if event.eventType() == blpapi.Event.RESPONSE:
                break
        elif event.eventType() == blpapi.Event.TIMEOUT:
            continue

    # =============================
    # STEP 4 — Save to CSV
    # =============================
    if records:
        df = pd.DataFrame(records)
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"🎉 Done! Static data saved to: {OUTPUT_PATH}")
    else:
        print("❌ Error: No static data returned.")

if __name__ == "__main__":
    main()