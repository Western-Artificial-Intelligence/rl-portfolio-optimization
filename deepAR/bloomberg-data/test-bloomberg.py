import blpapi
from blpapi import SessionOptions, Session

# Output file
output_file = "output.txt"

options = SessionOptions()
options.setServerHost("localhost")
options.setServerPort(8194)

session = Session(options)
if not session.start():
    print("Failed to start session.")
    exit()

if not session.openService("//blp/refdata"):
    print("Failed to open //blp/refdata service")
    exit()

service = session.getService("//blp/refdata")
request = service.createRequest("HistoricalDataRequest")

request.getElement("securities").appendValue("AAPL US Equity")
request.getElement("fields").appendValue("PX_LAST")

request.set("startDate", "20200101")
request.set("endDate", "20251231")
request.set("periodicitySelection", "DAILY")

print("Sending Request...")
session.sendRequest(request)

with open(output_file, "w", encoding="utf-8") as f:
    while True:
        event = session.nextEvent()
        for msg in event:
            f.write(str(msg) + "\n")   # Write each message to the file
        if event.eventType() == blpapi.Event.RESPONSE:
            break

print(f"Finished. Output saved to {output_file}")
