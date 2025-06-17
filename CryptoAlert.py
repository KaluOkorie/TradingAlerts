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

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]

def compute_tf_features(df, tf_label):
    feat = {}
    feat[f'ema_{tf_label}'] = ta.ema(df['close'], length=1).shift(1)
    feat[f'rsi_{tf_label}'] = ta.rsi(df['close'], length=14).shift(1)
    feat[f'atr_{tf_label}'] = ta.atr(df['high'], df['low'], df['close'], length=14).shift(1)
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    feat[f'adx_{tf_label}'] = adx['ADX_14'].shift(1)
    prior_high = df['high'].shift(1).rolling(20).max()
    feat[f'breakout_{tf_label}'] = (df['close'] > prior_high).astype(int)
    macd, macd_signal, macd_hist = compute_macd(df['volume'])
    feat[f'macd_{tf_label}'] = macd
    feat[f'macd_signal_{tf_label}'] = macd_signal
    feat[f'macd_hist_{tf_label}'] = macd_hist
    for k in feat:
        if hasattr(feat[k], "iloc"):
            feat[k] = feat[k].iloc[-1]
    return feat

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
            # STRICT: Only check the title for alias as a whole word
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

def decide_signal_and_confidence(feats, direction, rsi_threshold=45, adx_threshold=20):
    if direction == "BULLISH":
        ema_chain = feats['ema_1h'] > feats['ema_4h']
        rsi_fav = feats['rsi_4h'] > rsi_threshold
        adx_ok = feats['adx_4h'] > adx_threshold
        breakout_4h = feats['breakout_4h']
        macd_1h_pos = feats['macd_hist_1h'] > 0
        conditions = [ema_chain, rsi_fav, adx_ok, breakout_4h]
        entry = (sum(conditions) >= 3) and macd_1h_pos
        ema_conf = 1.0 if ema_chain else 0.0
        adx_conf = min(max((feats['adx_4h'] - adx_threshold) / 40.0, 0), 1)
        rsi_conf = min(max((feats['rsi_4h'] - rsi_threshold) / (100 - rsi_threshold), 0), 1)
        macd_conf = 1.0 if macd_1h_pos else 0.0
        confidence = (
            0.45 * ema_conf +
            0.25 * adx_conf +
            0.20 * rsi_conf +
            0.10 * macd_conf
        )
        print(f"[DEBUG] BULLISH: EMA_CHAIN={ema_chain}, RSI_FAV={rsi_fav}, ADX_OK={adx_ok}, BREAKOUT_4H={breakout_4h}, MACD_1h_POS={macd_1h_pos} | entry={entry} conf={confidence:.2f}")
    else:  # BEARISH
        ema_chain = feats['ema_1h'] < feats['ema_4h']
        rsi_fav = feats['rsi_4h'] < rsi_threshold
        adx_ok = feats['adx_4h'] > adx_threshold
        breakout_4h = feats['breakout_4h']
        macd_1h_neg = feats['macd_hist_1h'] < 0
        conditions = [ema_chain, rsi_fav, adx_ok, breakout_4h]
        entry = (sum(conditions) >= 3) and macd_1h_neg
        ema_conf = 1.0 if ema_chain else 0.0
        adx_conf = min(max((feats['adx_4h'] - adx_threshold) / 40.0, 0), 1)
        rsi_conf = min(max((rsi_threshold - feats['rsi_4h']) / rsi_threshold, 0), 1)
        macd_conf = 1.0 if macd_1h_neg else 0.0
        confidence = (
            0.45 * ema_conf +
            0.25 * adx_conf +
            0.20 * rsi_conf +
            0.10 * macd_conf
        )
        print(f"[DEBUG] BEARISH: EMA_CHAIN={ema_chain}, RSI_FAV={rsi_fav}, ADX_OK={adx_ok}, BREAKOUT_4H={breakout_4h}, MACD_1h_NEG={macd_1h_neg} | entry={entry} conf={confidence:.2f}")
    return entry, min(confidence, 1.0)

def estimate_trade_duration(price, atr):
    atr_ratio = atr / price
    if atr_ratio > 0.03:
        return round(10, 1)
    elif atr_ratio > 0.01:
        return round(40, 1)
    else:
        return round(120, 1)

def get_breakout_level(df4):
    return df4['high'].shift(1).rolling(20).max().iloc[-1]

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
            df4 = fetch_klines(s, '4h', 100)
            df1 = fetch_klines(s, '1h', 100)
            if df4.empty or df1.empty:
                print(f"[DEBUG] Empty df for {s}")
                continue
            feats = {}
            feats.update(compute_tf_features(df4, '4h'))
            feats.update(compute_tf_features(df1, '1h'))
            feats['ema_4h'] = feats['ema_4h']
            feats['ema_1h'] = feats['ema_1h']

            main_features = [
                'rsi_4h', 'ema_4h', 'ema_1h', 'adx_4h', 'atr_4h',
                'macd_hist_1h', 'macd_hist_4h'
            ]
            if any(np.isnan(feats.get(x, np.nan)) for x in main_features):
                print(f"[DEBUG] NaN feature for {s}")
                continue

            price = df4['close'].iloc[-1]
            atr = feats['atr_4h']
            breakout_level = get_breakout_level(df4)

            is_bull, conf_bull = decide_signal_and_confidence(feats, "BULLISH")
            if is_bull and conf_bull >= 0.6:
                tp = price + 2 * atr
                sl = price - 1.4 * atr
                duration = estimate_trade_duration(price, atr)
                bullish_signals.append(s)
                trade_details[s] = {
                    "direction": "BULLISH",
                    "confidence": conf_bull,
                    "breakout": breakout_level,
                    "price": price,
                    "tp": tp,
                    "sl": sl,
                    "duration": duration
                }

            is_bear, conf_bear = decide_signal_and_confidence(feats, "BEARISH")
            if is_bear and conf_bear >= 0.6:
                tp = price - 2 * atr
                sl = price + 1.4 * atr
                duration = estimate_trade_duration(price, atr)
                bearish_signals.append(s)
                trade_details[s] = {
                    "direction": "BEARISH",
                    "confidence": conf_bear,
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
        msg += "*No singal found *\n"

    print("[DEBUG] ------------------ FINAL SIGNAL SUMMARY ------------------")
    for sym in bullish_signals:
        print(f"BULLISH: {sym} | Conf: {trade_details[sym]['confidence']:.2f} | Price: {trade_details[sym]['price']:.4f}")
    for sym in bearish_signals:
        print(f"BEARISH: {sym} | Conf: {trade_details[sym]['confidence']:.2f} | Price: {trade_details[sym]['price']:.4f}")
    if not bullish_signals and not bearish_signals:
        print("*No singal found *")

    send_tel(msg)

if __name__ == '__main__':
    main()
