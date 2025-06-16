import os
import time
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
import pandas_ta as ta
from transformers import pipeline

# === CONFIGURATION ===
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

sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

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

def fetch_cryptocompare_news(symbol):
    url = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&api_key={API_KEY}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return []
        try:
            data = resp.json()
            news_data = data.get("Data", [])
        except Exception:
            return []
        aliases = COIN_ALIASES.get(symbol, [symbol])
        bull, bear = [], []
        for post in news_data:
            title = post.get("title", "")
            body = post.get("body", "")
            tags = post.get("tags", "")
            content = f"{tags} {title} {body}".lower()
            if any(alias.lower() in content for alias in aliases):
                text = title[:512]
                if text:
                    r = sentiment_pipe(text)[0]
                    if r['label'] == 'POSITIVE' and len(bull) < 2:
                        bull.append({'sentiment': 1, 'headline': title})
                    elif r['label'] == 'NEGATIVE' and len(bear) < 2:
                        bear.append({'sentiment': -1, 'headline': title})
                if len(bull) == 2 and len(bear) == 2:
                    break
        headlines = bull + bear
        return headlines
    except Exception:
        return []

def top_symbols(n):
    df = pd.DataFrame(requests.get(f"{BINANCE_API}/ticker/24hr").json())
    df['quoteVolume'] = pd.to_numeric(df['quoteVolume'], errors='coerce')
    usdt = df[df['symbol'].str.endswith('USDT')]
    top = usdt.nlargest(n, 'quoteVolume')['symbol'].tolist()
    return top

def decide_signal_and_confidence(feats, direction, rsi_threshold=45, adx_threshold=20):
    if direction == "BULLISH":
        ema_chain = feats['ema_15m'] > feats['ema_1h'] > feats['ema_4h']
        rsi_fav = feats['rsi_4h'] > rsi_threshold
        adx_ok = feats['adx_4h'] > adx_threshold
        macd_bull = (
            feats['macd_hist_15m'] > 0 and
            feats['macd_hist_1h'] > 0 and
            feats['macd_hist_4h'] > 0
        )
        entry = ema_chain and adx_ok and rsi_fav and macd_bull
        ema_conf = 1.0 if ema_chain else 0.0
        adx_conf = min(max((feats['adx_4h'] - adx_threshold) / 40.0, 0), 1)
        rsi_conf = min(max((feats['rsi_4h'] - rsi_threshold) / (100 - rsi_threshold), 0), 1)
        macd_15m_conf = 1.0 if feats['macd_hist_15m'] > 0 else 0.0
        macd_1h_conf = 1.0 if feats['macd_hist_1h'] > 0 else 0.0
        macd_4h_conf = 1.0 if feats['macd_hist_4h'] > 0 else 0.0
        macd_conf = (macd_15m_conf * 0.5 + macd_1h_conf * 0.3 + macd_4h_conf * 0.2)
        confidence = (
            0.40 * ema_conf +
            0.25 * adx_conf +
            0.20 * rsi_conf +
            0.15 * macd_conf
        )
    else:  # BEARISH
        ema_chain = feats['ema_15m'] < feats['ema_1h'] < feats['ema_4h']
        rsi_fav = feats['rsi_4h'] < rsi_threshold
        adx_ok = feats['adx_4h'] > adx_threshold
        macd_bear = (
            feats['macd_hist_15m'] < 0 and
            feats['macd_hist_1h'] < 0 and
            feats['macd_hist_4h'] < 0
        )
        entry = ema_chain and adx_ok and rsi_fav and macd_bear
        ema_conf = 1.0 if ema_chain else 0.0
        adx_conf = min(max((feats['adx_4h'] - adx_threshold) / 40.0, 0), 1)
        rsi_conf = min(max((rsi_threshold - feats['rsi_4h']) / rsi_threshold, 0), 1)
        macd_15m_conf = 1.0 if feats['macd_hist_15m'] < 0 else 0.0
        macd_1h_conf = 1.0 if feats['macd_hist_1h'] < 0 else 0.0
        macd_4h_conf = 1.0 if feats['macd_hist_4h'] < 0 else 0.0
        macd_conf = (macd_15m_conf * 0.5 + macd_1h_conf * 0.3 + macd_4h_conf * 0.2)
        confidence = (
            0.40 * ema_conf +
            0.25 * adx_conf +
            0.20 * rsi_conf +
            0.15 * macd_conf
        )
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

def main():
    now_utc = datetime.utcnow()
    now_bst = now_utc + timedelta(hours=1)
    date_str = now_bst.strftime("%A, %d %B %Y %H:%M BST")

    syms = top_symbols(TOP_N_SYMBOLS)
    bullish_signals = []
    bearish_signals = []
    trade_details = {}
    asset_sentiment = {}

    for s in syms:
        try:
            df4 = fetch_klines(s, '4h', 100)
            df1 = fetch_klines(s, '1h', 100)
            df15 = fetch_klines(s, '15m', 100)
            if df4.empty or df1.empty or df15.empty:
                continue
            feats = {}
            feats.update(compute_tf_features(df4, '4h'))
            feats.update(compute_tf_features(df1, '1h'))
            feats.update(compute_tf_features(df15, '15m'))
            feats['ema_4h'] = feats['ema_4h']
            feats['ema_1h'] = feats['ema_1h']
            feats['ema_15m'] = feats['ema_15m']

            main_features = [
                'rsi_4h', 'ema_4h', 'ema_1h', 'ema_15m', 'adx_4h', 'atr_4h',
                'macd_hist_15m', 'macd_hist_1h', 'macd_hist_4h'
            ]
            if any(np.isnan(feats.get(x, np.nan)) for x in main_features):
                continue

            price = df4['close'].iloc[-1]
            atr = feats['atr_4h']
            breakout_level = get_breakout_level(df4)

            is_bull, conf_bull = decide_signal_and_confidence(feats, "BULLISH")
            if is_bull:
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
            if is_bear:
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
        except Exception:
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
        msg += "No actionable signals found today.\n"

    # Ready for deployment: msg variable contains the summary for downstream use

if __name__ == '__main__':
    main()
