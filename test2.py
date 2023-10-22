#import okx.Trade as Trade
from okx.trade import Trade
from okx.market import Market
import pandas as pd
import pytz


tz = pytz.timezone("UTC")

flag = "1"  # live trading: 0, demo trading: 1
# Realtime API
""" api_key = "2f652d28-5519-465b-a9db-09abb5b1c0ef"
secret_key = "53EE60441E4ABE292B23E47547A1B49E"
passphrase = "8'gUc:Xg8m5kN;f" """
# Demo API
api_key = "1f39e790-6edb-4dba-a9fa-c234f0de1eca"
secret_key = "EC4320B104398AB65E0BC2BADD9778AE"
passphrase = "Z-$gP8Ur-ESE!&n"
tradeAPI = Trade(api_key, secret_key, passphrase, flag)
marketAPI = Market(api_key, secret_key, passphrase, flag)
""" # market order
result = tradeAPI.set_order(
    instId="BTC-USDT-SWAP",
    tdMode="isolated",
    side="buy",
    posSide="net",
    ordType="market",
    sz="100"
)
print(result)

if result["code"] == "0":
    print("Successful order request，order_id = ",result["data"][0]["ordId"])
else:
    print("Unsuccessful order request，error_code = ",result["data"][0]["sCode"], ", Error_message = ", result["data"][0]["sMsg"]) """

result = marketAPI.get_history_candles("BTC-USDT-SWAP", bar="1m")
if result["code"] == "0":
    df = pd.DataFrame(result['data'], columns=['datetime', 'open', 'high', 'low', 'close', 'volume', 'volumeCcy', 'volumevolCcyQuote', 'confirm'])
    #df['datetime'] = df['datetime'].astype(int)
    df['datetime'] = df['datetime'].astype(float)
    df['datetime'] = (df['datetime'] / 1000).astype(float)
    df['datetime'] = df['datetime'].astype(int)
    df['datetime'] = pd.to_datetime(df['datetime'], unit='s').dt.strftime("%Y-%m-%d %H:%M")
    print(df.head(5))
    print(df.tail(5))
else:
    print("NONE")
