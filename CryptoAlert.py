import os
import time
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
import pandas_ta as ta
from transformers import pipeline
import re

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
BINANCE_API = "https://api.binance.us/api/v3"
TOP_N_SYMBOLS = 50
API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY", "")

COIN_ALIASES = {
    "BTCUSDT": ["BTCUSDT", "BTC", "Bitcoin"],
    "ETHUSDT": ["ETHUSDT", "ETH", "Ethereum"],
    "XRPUSDT": ["XRPUSDT", "XRP", "Ripple"],
    "SOLUSDT": ["SOLUSDT", "SOL", "Solana"],
    "DOTUSDT": ["DOTUSDT", "DOT", "Polkadot"],
    "FLOKIUSDT": ["FLOKIUSDT", "FLOKI", "Floki Inu"],
    "SUIUSDT": ["SUIUSDT", "SUI", "Sui"],
    "DOGEUSDT": ["DOGEUSDT", "DOGE", "Dogecoin"],
    "PEPEUSDT": ["PEPEUSDT", "PEPE", "Pepe"],
    "USDCUSDT": ["USDCUSDT", "USDC", "USD Coin"],
    "FETUSDT": ["FETUSDT", "FET", "Fetch.ai"],
    "ADAUSDT": ["ADAUSDT", "ADA", "Cardano"],
    "BCHUSDT": ["BCHUSDT", "BCH", "Bitcoin Cash"],
    "GALAUSDT": ["GALAUSDT", "GALA", "Gala"],
    "BNBUSDT": ["BNBUSDT", "BNB", "Binance Coin"],
    "HYPEUSDT": ["HYPEUSDT", "HYPE", "Hype"],
    "SHIBUSDT": ["SHIBUSDT", "SHIB", "Shiba Inu"],
    "LTCUSDT": ["LTCUSDT", "LTC", "Litecoin"],
    "IOSTUSDT": ["IOSTUSDT", "IOST", "IOST"],
    "AVAXUSDT": ["AVAXUSDT", "AVAX", "Avalanche"],
    "ONEUSDT": ["ONEUSDT", "ONE", "Harmony"],
    "TRUMPUSDT": ["TRUMPUSDT", "TRUMP", "Trump Coin"],
    "SUSHIUSDT": ["SUSHIUSDT", "SUSHI", "SushiSwap"],
    "NEARUSDT": ["NEARUSDT", "NEAR", "NEAR Protocol"],
    "MKRUSDT": ["MKRUSDT", "MKR", "Maker"],
    "LINKUSDT": ["LINKUSDT", "LINK", "Chainlink"],
    "API3USDT": ["API3USDT", "API3", "API3"],
    "VETUSDT": ["VETUSDT", "VET", "VeChain"],
    "PNUTUSDT": ["PNUTUSDT", "PNUT", "Peanut"],
    "RVNUSDT": ["RVNUSDT", "RVN", "Ravencoin"],
    "HBARUSDT": ["HBARUSDT", "HBAR", "Hedera"],
    "AAVEUSDT": ["AAVEUSDT", "AAVE", "Aave"],
    "BANDUSDT": ["BANDUSDT", "BAND", "Band Protocol"],
    "XLMUSDT": ["XLMUSDT", "XLM", "Stellar"],
    "ATOMUSDT": ["ATOMUSDT", "ATOM", "Cosmos"],
    "VTHOUSDT": ["VTHOUSDT", "VTHO", "VeThor Token"],
    "POLUSDT": ["POLUSDT", "POL", "Polygon Ecosystem Token"],
    "PENGUUSDT": ["PENGUUSDT", "PENGU", "Penguin"],
    "BONKUSDT": ["BONKUSDT", "BONK", "Bonk"],
    "THETAUSDT": ["THETAUSDT", "THETA", "Theta"],
    "AIXBTUSDT": ["AIXBTUSDT", "AIXBT", "AI XBT"],
    "WIFUSDT": ["WIFUSDT", "WIF", "dogwifhat"],
    "YFIUSDT": ["YFIUSDT", "YFI", "Yearn.Finance"],
    "ACHUSDT": ["ACHUSDT", "ACH", "Alchemy Pay"],
    "UNIUSDT": ["UNIUSDT", "UNI", "Uniswap"],
    "ZILUSDT": ["ZILUSDT", "ZIL", "Zilliqa"],
    "IOTAUSDT": ["IOTAUSDT", "IOTA", "IOTA"],
    "RENUSDT": ["RENUSDT", "REN", "Ren"],
    "ENAUSDT": ["ENAUSDT", "ENA", "Ethena"],
    "RENDERUSDT": ["RENDERUSDT", "RNDR", "Render"],
}

# Robust sentiment pipeline loading
try:
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
except Exception as e:
    print(f"[DEBUG] Sentiment model load failed: {e}")
    sentiment_pipe = None

def fetch_klines(symbol, interval, limit):
    url = f"{BINANCE_API}/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            print(f"[DEBUG] fetch_klines: HTTP {resp.status_code} for {symbol} {interval}")
            return pd.DataFrame()
        data = resp.json()
        if not isinstance(data, list) or not data:
            print(f"[DEBUG] fetch_klines: No data for {symbol} {interval}")
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms').dt.tz_localize(None)
        return df
    except Exception as e:
        print(f"[DEBUG] fetch_klines Exception for {symbol} {interval}: {e}")
        return pd.DataFrame()

def fetch_cryptocompare_news(symbol):
    url = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&api_key={API_KEY}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            print(f"[DEBUG] News fetch HTTP error {resp.status_code} for {symbol}")
            return []
        try:
            data = resp.json()
            news_data = data.get("Data", [])
        except Exception as jserr:
            print(f"[DEBUG] News fetch JSON error for {symbol}: {jserr}")
            return []
        aliases = COIN_ALIASES.get(symbol, [symbol])
        alias_pattern = r'\b(' + '|'.join(re.escape(alias) for alias in aliases) + r')\b'
        bull, bear = [], []
        for post in news_data:
            title = post.get("title", "")
            if re.search(alias_pattern, title, flags=re.IGNORECASE):
                text = title[:512]
                if text and sentiment_pipe is not None:
                    try:
                        r = sentiment_pipe(text)[0]
                        if r['label'] == 'POSITIVE' and len(bull) < 2:
                            bull.append({'sentiment': 1, 'headline': title})
                        elif r['label'] == 'NEGATIVE' and len(bear) < 2:
                            bear.append({'sentiment': -1, 'headline': title})
                        if len(bull) == 2 and len(bear) == 2:
                            break
                    except Exception as sentiment_error:
                        print(f"[DEBUG] Sentiment analysis error: {sentiment_error}")
                        continue
            if len(bull) == 2 and len(bear) == 2:
                break
        headlines = bull + bear
        print(f"[DEBUG] STRICT News for {symbol}: {headlines}")
        return headlines
    except Exception as e:
        print(f"[DEBUG] News fetch error for {symbol}: {e}")
        return []

def top_symbols(n):
    try:
        df = pd.DataFrame(requests.get(f"{BINANCE_API}/ticker/24hr").json())
        df['quoteVolume'] = pd.to_numeric(df['quoteVolume'], errors='coerce')
        usdt = df[df['symbol'].str.endswith('USDT')]
        top = usdt.nlargest(n, 'quoteVolume')['symbol'].tolist()
        print(f"[DEBUG] Top symbols: {top[:10]} ...")
        return top
    except Exception as e:
        print(f"[DEBUG] Error fetching top symbols: {e}")
        return []

def get_asset_pair(symbol):
    if symbol.endswith("USDT"):
        base = symbol[:-4]
        quote = "USDT"
        return f"{base}/{quote}", base
    elif symbol.endswith("USD"):
        base = symbol[:-3]
        quote = "USD"
        return f"{base}/{quote}", base
    else:
        base = symbol[:-3]
        quote = symbol[-3:]
        return f"{base}/{quote}", base

def format_signal(sym, details, news_headlines):
    asset_pair, base = get_asset_pair(sym)
    conf_percent = int(round(details['confidence'] * 100))
    b_count = sum(1 for n in news_headlines if n['sentiment'] > 0)
    r_count = sum(1 for n in news_headlines if n['sentiment'] < 0)
    news_str = ""
    if news_headlines:
        news_str = f"📊News for asset B={b_count}, R={r_count}\n"
        for n in news_headlines:
            prefix = "🟢" if n['sentiment'] > 0 else "🔴"
            news_str += f"{prefix} {n['headline']}\n"
    return (
        f"🚀 *{details['direction']} Signal*\n"
        f"📈 Asset: *{base} ({asset_pair})*\n"
        f"🔍 Confidence: {conf_percent}%\n"
        f"🔺 *Breakout Above*: ${details['breakout']:.4f}\n"
        f"💰 Current Price: ${details['price']:.4f}\n"
        f"⏱️ Trade Duration: {details['duration']} minutes\n"
        f"🛑 Stop Loss: ${details['sl']:.4f}\n"
        f"🎯 Take Profit: ${details['tp']:.4f}\n"
        f"{news_str}"
    )

def send_tel(msg):
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                data={'chat_id': TELEGRAM_CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'}
            )
            print(f"[DEBUG] Telegram send status code: {r.status_code}")
            if r.status_code != 200:
                print(f"[DEBUG] Telegram error: {r.text}")
        except Exception as e:
            print(f"[DEBUG] Telegram send exception: {e}")

def apply_indicators(df):
    df['rsi_fast'] = ta.rsi(df['close'], length=7)
    df['rsi_slow'] = ta.rsi(df['close'], length=14)
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx'] = adx['ADX_14']
    macd = ta.macd(df['close'])
    df['macd_line'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    ichimoku, _ = ta.ichimoku(df['high'], df['low'], df['close'])
    df['tenkan'] = ichimoku['ITS_9']
    df['kijun'] = ichimoku['IKS_26']
    bbands = ta.bbands(df['close'], length=20)
    df['bb_upper'] = bbands['BBU_20_2.0']
    df['bb_middle'] = bbands['BBM_20_2.0']
    df['bb_lower'] = bbands['BBL_20_2.0']
    df['ma'] = df['close'].rolling(window=20).mean()
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df = halftrend(df, length=10, multiplier=2)
    return df

def halftrend(df, length=10, multiplier=2):
    hl2 = (df['high'] + df['low']) / 2
    atr = ta.atr(df['high'], df['low'], df['close'], length=length)
    trend = pd.Series(index=df.index, dtype='float64')
    up = hl2 - multiplier * atr
    dn = hl2 + multiplier * atr
    trend.iloc[0] = 0
    for i in range(1, len(df)):
        if hl2.iloc[i] > up.iloc[i-1]:
            trend.iloc[i] = 1
        elif hl2.iloc[i] < dn.iloc[i-1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]
    df['halftrend'] = trend
    return df

def signal_generator(df):
    df['signal'] = 0
    for i in range(1, len(df)):
        if (
            df['rsi_fast'][i] > 50 and
            df['adx'][i] > 25 and
            df['macd_line'][i] > 0 and
            df['rsi_fast'][i] > df['rsi_slow'][i] and
            df['tenkan'][i] > df['kijun'][i] and
            df['close'][i] > df['ma'][i] and
            df['close'][i-1] <= df['ma'][i-1] and
            df['halftrend'][i] == 1
        ):
            df.loc[i, 'signal'] = 1  # Buy
        elif (
            df['rsi_fast'][i] < 50 and
            df['adx'][i] > 25 and
            df['macd_line'][i] < 0 and
            df['rsi_fast'][i] < df['rsi_slow'][i] and
            df['tenkan'][i] < df['kijun'][i] and
            df['close'][i] < df['ma'][i] and
            df['close'][i-1] >= df['ma'][i-1] and
            df['halftrend'][i] == -1
        ):
            df.loc[i, 'signal'] = -1  # Sell
    return df

def compute_dynamic_confidence(latest):
    checks = [
        latest['rsi_fast'] > 50,
        latest['adx'] > 25,
        latest['macd_line'] > 0,
        latest['rsi_fast'] > latest['rsi_slow'],
        latest['tenkan'] > latest['kijun'],
        latest['close'] > latest['ma'],
        latest['halftrend'] == 1
    ]
    score = sum(checks)
    total = len(checks)
    confidence = score / total
    return confidence

def compute_dynamic_confidence_bear(latest):
    checks = [
        latest['rsi_fast'] < 50,
        latest['adx'] > 25,
        latest['macd_line'] < 0,
        latest['rsi_fast'] < latest['rsi_slow'],
        latest['tenkan'] < latest['kijun'],
        latest['close'] < latest['ma'],
        latest['halftrend'] == -1
    ]
    score = sum(checks)
    total = len(checks)
    confidence = score / total
    return confidence

def main():
    now_utc = datetime.utcnow()
    now_bst = now_utc + timedelta(hours=1)
    date_str = now_bst.strftime("%A, %d %B %Y %H:%M BST")
    print(f"[DEBUG] Current time: {date_str}")

    syms = top_symbols(TOP_N_SYMBOLS)
    bullish_signals = []
    bearish_signals = []
    trade_details = {}
    asset_sentiment = {}

    for s in syms:
        try:
            print(f"[DEBUG] Processing {s}")
            df = fetch_klines(s, '1h', 500)
            if df.empty:
                print(f"[DEBUG] Empty df for {s}")
                continue
            df = apply_indicators(df)
            df = signal_generator(df)
            two_weeks_ago = pd.Timestamp.utcnow() - pd.Timedelta(days=14)
            df = df[df['timestamp'] >= two_weeks_ago].reset_index(drop=True)
            if df.empty:
                print(f"[DEBUG] No recent data for {s}")
                continue
            latest = df.iloc[-1]
            price = latest['close']
            atr = latest['atr']
            breakout_level = df['high'].shift(1).rolling(20).max().iloc[-1]
            direction = None
            confidence = 0
            tp = sl = duration = None

            if latest['signal'] == 1:
                confidence = compute_dynamic_confidence(latest)
                if confidence > 0.65:
                    direction = "BULLISH"
                    tp = price + 2 * atr
                    sl = price - 1.4 * atr
                    duration = 60
                    bullish_signals.append(s)
            elif latest['signal'] == -1:
                confidence = compute_dynamic_confidence_bear(latest)
                if confidence > 0.65:
                    direction = "BEARISH"
                    tp = price - 2 * atr
                    sl = price + 1.4 * atr
                    duration = 60
                    bearish_signals.append(s)

            if direction:
                trade_details[s] = {
                    "direction": direction,
                    "confidence": confidence,
                    "breakout": breakout_level,
                    "price": price,
                    "tp": tp,
                    "sl": sl,
                    "duration": duration
                }
            news_headlines = fetch_cryptocompare_news(s)
            asset_sentiment[s] = news_headlines
        except Exception as e:
            print(f"[DEBUG] Error processing {s}: {e}")
            continue
    header = f"*🚀 DAily CRYPTO TRADE SIGNAL  — {date_str}*"
    msg = header + "\n"

    if bullish_signals:
        for sym in bullish_signals[:3]:
            msg += format_signal(sym, trade_details[sym], asset_sentiment.get(sym, [])) + "\n"
    if bearish_signals:
        for sym in bearish_signals[:3]:
            msg += format_signal(sym, trade_details[sym], asset_sentiment.get(sym, [])) + "\n"
    if not bullish_signals and not bearish_signals:
        msg += "*No signal found*\n"

    print("[DEBUG] ------------------ FINAL SIGNAL SUMMARY ------------------")
    for sym in bullish_signals:
        print(f"BULLISH: {sym} | Conf: {trade_details[sym]['confidence']:.2f} | Price: {trade_details[sym]['price']:.4f}")
    for sym in bearish_signals:
        print(f"BEARISH: {sym} | Conf: {trade_details[sym]['confidence']:.2f} | Price: {trade_details[sym]['price']:.4f}")
    if not bullish_signals and not bearish_signals:
        print("*No signal found*")

    send_tel(msg)

if __name__ == '__main__':
    main()
