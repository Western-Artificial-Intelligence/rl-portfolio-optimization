import blpapi
import pandas as pd
from datetime import datetime

def main():
    # 1. Setup Bloomberg Session
    options = blpapi.SessionOptions()
    options.setServerHost("localhost")
    options.setServerPort(8194)
    
    session = blpapi.Session(options)
    if not session.start():
        print("❌ Failed to start session. Is the Bloomberg Terminal logged in?")
        return

    if not session.openService("//blp/refdata"):
        print("❌ Failed to open //blp/refdata")
        return

    refDataService = session.getService("//blp/refdata")

    # ---------------------------------------------------------
    # STEP 1: Get NASDAQ-100 Constituents
    # ---------------------------------------------------------
    print("Requesting NASDAQ-100 member list...")
    
    # We use ReferenceDataRequest for current index members
    index_req = refDataService.createRequest("ReferenceDataRequest")
    index_req.append("securities", "NDX Index")
    index_req.append("fields", "INDX_MEMBERS") 
    
    # Send request
    session.sendRequest(index_req)

    tickers = []
    
    # Process Index Response
    while True:
        event = session.nextEvent(500) # 500ms timeout
        if event.eventType() == blpapi.Event.RESPONSE:
            for msg in event:
                if msg.hasElement("securityData"):
                    sec_data_array = msg.getElement("securityData")
                    for i in range(sec_data_array.numValues()):
                        sec_data = sec_data_array.getValue(i)
                        field_data = sec_data.getElement("fieldData")
                        
                        if field_data.hasElement("INDX_MEMBERS"):
                            member_array = field_data.getElement("INDX_MEMBERS")
                            for j in range(member_array.numValues()):
                                member_data = member_array.getValue(j)
                                
                                # --- FIX: ROBUST FIELD CHECKING ---
                                t = None
                                if member_data.hasElement("Member Ticker and Exchange Code"):
                                    t = member_data.getElementAsString("Member Ticker and Exchange Code")
                                elif member_data.hasElement("Member Ticker"):
                                    t = member_data.getElementAsString("Member Ticker")
                                else:
                                    # If unknown schema, print it to debug
                                    print(f"⚠️ Warning: Could not find ticker in element: {member_data}")
                                    continue
                                
                                if t:
                                    tickers.append(t)
            break
        elif event.eventType() == blpapi.Event.TIMEOUT:
            # Keep waiting if we haven't found the response yet
            continue

    print(f"Found {len(tickers)} NASDAQ-100 constituents.")
    if len(tickers) > 0:
        print(f"Sample raw tickers: {tickers[:3]}")
    else:
        print("❌ No tickers found. Exiting.")
        return

    # ---------------------------------------------------------
    # STEP 2: Normalize Tickers (Critical for History)
    # ---------------------------------------------------------
    # Transform 'AAPL UW' -> 'AAPL US Equity'
    clean_tickers = [f"{t.split()[0]} US Equity" for t in tickers]
    
    print(f"Normalized tickers: {clean_tickers[:3]}")

    # ---------------------------------------------------------
    # STEP 3: Request Price History (Chunked)
    # ---------------------------------------------------------
    print("\nRequesting price history from Bloomberg...")
    
    all_data = []
    BATCH_SIZE = 50  # Prevent API overload
    
    # Define fields and dates
    FIELDS = ["PX_LAST", "PX_OPEN", "PX_HIGH", "PX_LOW", "PX_VOLUME"]
    START_DATE = "20200101"
    END_DATE = datetime.today().strftime('%Y%m%d')

    for i in range(0, len(clean_tickers), BATCH_SIZE):
        batch = clean_tickers[i : i + BATCH_SIZE]
        print(f"Processing batch {i} to {i+len(batch)}...")

        hist_req = refDataService.createRequest("HistoricalDataRequest")
        
        for t in batch:
            hist_req.getElement("securities").appendValue(t)
        
        for f in FIELDS:
            hist_req.getElement("fields").appendValue(f)
            
        hist_req.set("periodicityAdjustment", "ACTUAL")
        hist_req.set("periodicitySelection", "DAILY")
        hist_req.set("startDate", START_DATE)
        hist_req.set("endDate", END_DATE)

        session.sendRequest(hist_req)

        # Process Historical Response for this batch
        while True:
            event = session.nextEvent(2000)
            
            # Catch errors/partial responses
            if event.eventType() in [blpapi.Event.RESPONSE, blpapi.Event.PARTIAL_RESPONSE]:
                for msg in event:
                    # Check for request errors
                    if msg.hasElement("responseError"):
                        print(f"⚠️ Error: {msg.getElement('responseError')}")
                        continue

                    # Parse security data
                    if msg.hasElement("securityData"):
                        sec_node = msg.getElement("securityData")
                        ticker = sec_node.getElementAsString("security")
                        
                        if sec_node.hasElement("fieldData"):
                            field_data_array = sec_node.getElement("fieldData")
                            
                            for k in range(field_data_array.numValues()):
                                point = field_data_array.getValue(k)
                                date_str = point.getElementAsString("date")
                                
                                # Build row dictionary
                                row = {
                                    "ticker": ticker,
                                    "date": date_str
                                }
                                # Extract requested fields safely
                                for f in FIELDS:
                                    if point.hasElement(f):
                                        row[f] = point.getElementAsFloat(f)
                                    else:
                                        row[f] = None
                                
                                all_data.append(row)

                if event.eventType() == blpapi.Event.RESPONSE:
                    break  # Batch done
            
            elif event.eventType() == blpapi.Event.TIMEOUT:
                pass

    # ---------------------------------------------------------
    # STEP 4: Save to CSV
    # ---------------------------------------------------------
    if all_data:
        df = pd.DataFrame(all_data)
        output_file = "data/nasdaq100_prices.csv"
        df.to_csv(output_file, index=False)
        print(f"\n🎉 Done! Saved {len(df)} rows to: {output_file}")
    else:
        print("\n❌ Error: No price data was returned. Check your terminal connection.")

if __name__ == "__main__":
    main()