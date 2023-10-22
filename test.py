import okx.MarketData as MarketData
import okx.Account as Account
import okx.Trade as Trade
flag = "1"  # live trading: 0, demo trading: 1


accountAPI = Account.AccountAPI(api_key, secret_key, passphrase, False, flag)
marketDataAPI = MarketData.MarketAPI(flag = flag)
tradeAPI = Trade.TradeAPI()
result = marketDataAPI.get_tickers(instType = "SWAP")
print(result)

import okx.Account as Account
flag = "1"  # live trading:0, demo trading: 1


result = accountAPI.get_account_balance()
print(result)


# market order
result = tradeAPI.place_order(
    instId = "BTC-USDT-SWAP",
    tdMode = "isolated",
    side = "buy",
    posSide = "net",
    ordType = "market",
    sz = "100"
)
print(result)

if result["code"] == "0":
    print("Successful order request，order_id = ",result["data"][0]["ordId"])
else:
    print("Unsuccessful order request，error_code = ",result["data"][0]["sCode"], ", Error_message = ", result["data"][0]["sMsg"])