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
from datetime import datetime

def main():
    # 1. Setup Bloomberg Session
    options = blpapi.SessionOptions()
    options.setServerHost("localhost")
    options.setServerPort(8194)
    
    session = blpapi.Session(options)
    if not session.start():
        print("❌ Failed to start session.")
        return

    if not session.openService("//blp/refdata"):
        print("❌ Failed to open //blp/refdata")
        return

    refDataService = session.getService("//blp/refdata")

    # ---------------------------------------------------------
    # STEP 1: Get S&P 500 Constituents
    # ---------------------------------------------------------
    print("Requesting S&P 500 (SPX Index) member list...")
    
    index_req = refDataService.createRequest("ReferenceDataRequest")
    index_req.append("securities", "SPX Index")
    index_req.append("fields", "INDX_MEMBERS") 
    
    session.sendRequest(index_req)

    tickers = []
    
    while True:
        event = session.nextEvent(500)
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
                                
                                # --- FIX 1: Robust Field Checking ---
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

    print(f"Found {len(tickers)} S&P 500 constituents.")

    # --- SAFETY CHECK ---
    if not tickers:
        print("❌ Error: No tickers found. Exiting to prevent crash.")
        return

    # ---------------------------------------------------------
    # STEP 2: Normalize Tickers
    # ---------------------------------------------------------
    # Ensure all tickers end in ' US Equity'
    clean_tickers = [f"{t.split()[0]} US Equity" for t in tickers]
    
    print(f"Sample normalized tickers: {clean_tickers[:3]}")

    # ---------------------------------------------------------
    # STEP 3: Request Price History (Chunked)
    # ---------------------------------------------------------
    # S&P 500 is too large for one request. We MUST chunk it.
    print("\nRequesting price history from Bloomberg...")
    
    all_data = []
    BATCH_SIZE = 50 
    
    FIELDS = ["PX_LAST", "PX_OPEN", "PX_HIGH", "PX_LOW", "PX_VOLUME"]
    START_DATE = "20180101"
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

        while True:
            event = session.nextEvent(2000)
            
            if event.eventType() in [blpapi.Event.RESPONSE, blpapi.Event.PARTIAL_RESPONSE]:
                for msg in event:
                    # --- FIX 2: Check for Errors before parsing ---
                    if msg.hasElement("responseError"):
                        print(f"⚠️ API Error: {msg.getElement('responseError')}")
                        continue

                    if msg.hasElement("securityData"):
                        sec_node = msg.getElement("securityData")
                        ticker = sec_node.getElementAsString("security")
                        
                        if sec_node.hasElement("fieldData"):
                            field_data_array = sec_node.getElement("fieldData")
                            
                            for k in range(field_data_array.numValues()):
                                point = field_data_array.getValue(k)
                                date_str = point.getElementAsString("date")
                                
                                row = {"ticker": ticker, "date": date_str}
                                
                                for f in FIELDS:
                                    if point.hasElement(f):
                                        row[f] = point.getElementAsFloat(f)
                                    else:
                                        row[f] = None
                                
                                all_data.append(row)

                if event.eventType() == blpapi.Event.RESPONSE:
                    break 
            
            elif event.eventType() == blpapi.Event.TIMEOUT:
                pass

    # ---------------------------------------------------------
    # STEP 4: Save to CSV
    # ---------------------------------------------------------
    if all_data:
        df = pd.DataFrame(all_data)
        output_file = "data/sp500_prices.csv"
        df.to_csv(output_file, index=False)
        print(f"\n🎉 Done! Saved {len(df)} rows to: {output_file}")
    else:
        print("\n❌ Error: No price data was returned.")

if __name__ == "__main__":
    main()