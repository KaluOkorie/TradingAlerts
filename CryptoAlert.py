import os
import ccxt
import requests
import pandas as pd
from datetime import datetime, timedelta

# ─── CONFIG ────────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
VOLUME_THRESH      = 1_000_000       # $1M 24h volume
MARKETCAP_THRESH   = 50_000_000      # $50M market cap
TIMEFRAME          = '4h'            # Candle timeframe

# ─── UTILS ─────────────────────────────────────────────────────────────────────
exchange = ccxt.kraken({ 'enableRateLimit': True })

def send_telegram(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = { 'chat_id': TELEGRAM_CHAT_ID, 'text': text }
    requests.post(url, data=payload)

# ─── 1) FILTER PAIRS BY VOLUME & MARKETCAP ────────────────────────────────────
exchange.load_markets()
tickers = exchange.fetch_tickers([s for s in exchange.symbols if s.endswith('/USD') or s.endswith('/USDT')])
high_vol = { sym for sym, data in tickers.items() if data.get('quoteVolume', 0) >= VOLUME_THRESH }

cg = requests.get(
    "https://api.coingecko.com/api/v3/coins/markets",
    params={ 'vs_currency': 'usd', 'order':'market_cap_desc', 'per_page':250, 'page':1 }
).json()
market_cap_ok = { f"{item['symbol'].upper()}/USD" for item in cg if item['market_cap'] >= MARKETCAP_THRESH }

candidates = sorted(high_vol & market_cap_ok)

# ─── 2) FETCH & SIGNAL TEST ────────────────────────────────────────────────────

def fetch_ohlcv(symbol: str):
    since = exchange.parse8601((datetime.utcnow() - timedelta(days=21)).isoformat())
    ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, since)
    df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','vol'])
    df.set_index(pd.to_datetime(df['ts'], unit='ms'), inplace=True)
    return df[['open','high','low','close','vol']]


def compute_signal(df: pd.DataFrame):
    if len(df) < 30: return None
    df['EMA9']  = df['close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean()
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    df['RSI'] = 100 - (100 / (1 + (gain.ewm(span=14).mean() / loss.ewm(span=14).mean()))).fillna(50)
    macd = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    signal = macd.ewm(span=9).mean()
    df['MACD'] = macd; df['Signal'] = signal
    last, prev = df.iloc[-1], df.iloc[-2]
    bullish = all([
        last.EMA9  > last.EMA21,
        last.RSI   > 50,
        last.MACD  > last.Signal,
        last.close > prev.close,
        last.close > last.open
    ])
    bearish = all([
        last.EMA9  < last.EMA21,
        last.RSI   < 50,
        last.MACD  < last.Signal,
        last.close < prev.close,
        last.close < last.open
    ])
    if bullish:  return 'Consider *BUY*'
    if bearish: return 'Consider *SELL*'
    return None

# ─── 3) RUN & NOTIFY ───────────────────────────────────────────────────────────

alerts = []
for sym in candidates:
    try:
        df = fetch_ohlcv(sym)
        action = compute_signal(df)
        if action:
            alerts.append(f"🔔 {sym}: {action}")
    except Exception:
        continue

if alerts:
    date = datetime.now().strftime('%A, %d %B %Y %H:%M UTC')
    message = f"*Crypto Alerts — {date}*\n" + '\n'.join(alerts)
    send_telegram(message)
