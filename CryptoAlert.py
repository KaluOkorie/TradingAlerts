import os
import ccxt
import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz

# ─── CONFIG ────────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
VOLUME_THRESH      = 1_000_000       # $1M 24h volume
MARKETCAP_THRESH   = 50_000_000      # $50M market cap
TIMEFRAME          = '4h'

# ─── UTILS ─────────────────────────────────────────────────────────────────────
exchange = ccxt.kraken({ 'enableRateLimit': True })

def send_telegram(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = { 'chat_id': TELEGRAM_CHAT_ID, 'text': text, 'parse_mode': 'Markdown' }
    requests.post(url, data=payload)

def get_uk_time():
    now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
    uk_time = now_utc.astimezone(pytz.timezone("Europe/London"))
    return uk_time.strftime('%A, %d %B %Y %H:%M %Z')

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

def compute_signal(df: pd.DataFrame, sym: str):
    if len(df) < 30: return None

    df['EMA9']  = df['close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean()

    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    df['RSI'] = 100 - (100 / (1 + (gain.ewm(span=14).mean() / loss.ewm(span=14).mean()))).fillna(50)

    df['MACD']   = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['Signal'] = df['MACD'].ewm(span=9).mean()

    last  = df.iloc[-1]
    prev  = df.iloc[-2]
    prev3 = df.iloc[-4:-1]

    recent_high = df['high'].iloc[-6:-2].max()
    recent_low  = df['low'].iloc[-6:-2].min()

    # ─── Breakout & Retest Logic ───────────────────────────────────────────────
    breakout_up    = last.close > recent_high
    breakout_down  = last.close < recent_low

    retest_up      = prev3['low'].min() <= recent_high * 1.01 and last.close > recent_high
    retest_down    = prev3['high'].max() >= recent_low  * 0.99 and last.close < recent_low

    bullish = all([
        last.EMA9  > last.EMA21,
        last.RSI   > 50,
        last.MACD  > last.Signal,
        last.close > prev.close,
        last.close > last.open,
        breakout_up,
        retest_up
    ])

    bearish = all([
        last.EMA9  < last.EMA21,
        last.RSI   < 50,
        last.MACD  < last.Signal,
        last.close < prev.close,
        last.close < last.open,
        breakout_down,
        retest_down
    ])

    if bullish:
        return f"🔼 *{sym}* — Consider *BUY*\nBreakout + Retest of resistance at `${recent_high:.2f}` → Price: `${last.close:.2f}`"
    if bearish:
        return f"🔽 *{sym}* — Consider *SELL*\nBreakdown + Retest of support at `${recent_low:.2f}` → Price: `${last.close:.2f}`"
    return None

# ─── 3) RUN & NOTIFY ───────────────────────────────────────────────────────────

alerts = []
for sym in candidates:
    try:
        df = fetch_ohlcv(sym)
        result = compute_signal(df, sym)
        if result:
            alerts.append(result)
    except Exception:
        continue

if alerts:
    header = f"*Crypto Alerts — {get_uk_time()}*"
    message = header + "\n\n" + "\n\n".join(alerts)
    send_telegram(message)
