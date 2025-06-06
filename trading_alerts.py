import os
from datetime import datetime
import pytz
import yfinance as yf
import requests
import pandas as pd

# ─── TELEGRAM CONFIG ──────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

# ─── ASSETS LIST ──────────────────────────────────────────────────────────────
# Forex pairs, major indices, gold futures (replacing XAU=X), and top ETFs/commodities.
ASSETS = {
    # ─── Forex Pairs ───────────────────────────────
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "USDJPY=X": "USD/JPY",
    "AUDUSD=X": "AUD/USD",
    "NZDUSD=X": "NZD/USD",
    "USDCAD=X": "USD/CAD",
    "USDCHF=X": "USD/CHF",
    "EURJPY=X": "EUR/JPY",
    "EURGBP=X": "EUR/GBP",

    # ─── Global Indices ────────────────────────────
    "^GSPC":     "S&P 500",
    "^IXIC":     "NASDAQ Composite",
    "^DJI":      "Dow Jones Industrial",
    "^GDAXI":    "DAX (Germany)",
    "^FTSE":     "FTSE 100 (UK)",
    "^STOXX50E": "Euro Stoxx 50",
    "^N225":     "Nikkei 225 (Japan)",
    "^HSI":      "Hang Seng Index (Hong Kong)",
    "^AXJO":     "ASX 200 (Australia)",
    "^BVSP":     "Bovespa (Brazil)",

    # ─── Commodities / Metals / Oil ────────────────
    "GC=F":  "Gold Futures",
    "SI=F":  "Silver Futures",
    "CL=F":  "Crude Oil (WTI)",
    "NG=F":  "Natural Gas",
    "ZC=F":  "Corn Futures",
    "ZS=F":  "Soybean Futures",
    "HG=F":  "Copper Futures",
    "PL=F":  "Platinum Futures",

    # ─── Popular ETFs ──────────────────────────────
    "SPY":  "SPDR S&P 500 ETF",
    "QQQ":  "Invesco QQQ Trust",
    "IWM":  "iShares Russell 2000 ETF",
    "VTI":  "Vanguard Total Stock Market ETF",
    "VEA":  "Vanguard FTSE Developed Markets ETF",
    "AGG":  "iShares Core US Aggregate Bond ETF",
    "XLF":  "Financial Select Sector ETF",
    "XLV":  "Health Care Select Sector ETF",

    # ─── Blue-Chip Stocks ──────────────────────────
    "AAPL":   "Apple",
    "MSFT":   "Microsoft",
    "GOOGL":  "Alphabet (Google)",
    "AMZN":   "Amazon",
    "TSLA":   "Tesla",
    "JNJ":    "Johnson & Johnson",
    "KO":     "Coca-Cola",
    "WMT":    "Walmart",
    "V":      "Visa",
    "PG":     "Procter & Gamble",
    "NVDA":   "NVIDIA",
    "META":   "Meta Platforms (Facebook)"}

# ─── INDICATOR PARAMETERS ────────────────────────────────────────────────────
SHORT_EMA  = 9
LONG_EMA   = 21
RSI_PERIOD = 14

def send_telegram_message(text: str) -> bool:
    """
    Sends a Markdown-formatted Telegram message to the configured chat.
    Returns True if status_code == 200, else False.
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id":   TELEGRAM_CHAT_ID,
        "text":      text,
        "parse_mode": "Markdown",
    }
    try:
        resp = requests.post(url, data=payload, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print("Telegram send error:", e)
        return False

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with at least 'Open' and 'Close', add:
      - EMA9   (on Close)
      - EMA21  (on Close)
      - RSI(14)
      - MACD (12,26) and Signal line (9)
    """
    # 1) EMAs on Close
    df['EMA9']  = df['Close'].ewm(span=SHORT_EMA, adjust=False).mean()
    df['EMA21'] = df['Close'].ewm(span=LONG_EMA, adjust=False).mean()

    # 2) RSI(14) on Close
    delta    = df['Close'].diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=RSI_PERIOD).mean()
    avg_loss = loss.rolling(window=RSI_PERIOD).mean()
    rs       = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 3) MACD (12,26) and Signal (9) on Close
    df['MACD']   = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    return df

def analyze_asset(symbol: str, name: str) -> str | None:
    """
    1. Downloads the last 7 days of 60m data for `symbol`.
    2. Computes EMA9, EMA21, RSI(14), MACD, Signal(9).
    3. If all 5 bullish conditions hold, return a LONG message.
       If all 5 bearish conditions hold, return a SHORT message.
       Otherwise, return None.
    """
    interval = "60m"
    period   = "7d"

    try:
        df = yf.download(symbol, interval=interval, period=period, auto_adjust=True)
        if df.empty:
            return None

        df = calculate_indicators(df)

        # Drop any rows where one of the indicators is NaN
        try:
            df = df.dropna(subset=['EMA9', 'EMA21', 'RSI', 'MACD', 'Signal'])
        except KeyError:
            # If these columns don't exist for some reason, skip
            return None

        if len(df) < 2:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2]

        # Five-condition checks (cast to bool() to ensure Python bool)
        bullish = all([
            bool(last['EMA9']  > last['EMA21']),   # 1) EMA9 above EMA21
            bool(last['RSI']   > 50),              # 2) RSI above 50
            bool(last['MACD']  > last['Signal']),  # 3) MACD above Signal
            bool(last['Close'] > prev['Close']),   # 4) Close > previous Close
            bool(last['Close'] > last['Open'])     # 5) Close > Open
        ])
        bearish = all([
            bool(last['EMA9']  < last['EMA21']),   # 1) EMA9 below EMA21
            bool(last['RSI']   < 50),              # 2) RSI below 50
            bool(last['MACD']  < last['Signal']),  # 3) MACD below Signal
            bool(last['Close'] < prev['Close']),   # 4) Close < previous Close
            bool(last['Close'] < last['Open'])     # 5) Close < Open
        ])

        if bullish:
            return f"🟢 {name}: Consider going *LONG* — All 5 bullish conditions met."
        elif bearish:
            return f"🔴 {name}: Consider going *SHORT* — All 5 bearish conditions met."
        else:
            return None

    except Exception as e:
        # Skip this asset if any unexpected error occurs
        print(f"Error analyzing {name}: {e}")
        return None

def main():
    """
    1. Iterate through every symbol in ASSETS (all 60m timeframe).
    2. Collect any LONG/SHORT messages (skip None).
    3. If at least one signal exists, send a single Telegram message labeled
       with the current London timestamp (YYYY-MM-DD HH:MM BST/GMT).
    """
    print("Running trading signal analysis…")

    signals = []
    for symbol, name in ASSETS.items():
        msg = analyze_asset(symbol, name)
        if msg:
            signals.append(msg)

    if not signals:
        print("No bullish/bearish signals detected at this time; exiting.")
        return

    # Build a timestamp header in Europe/London timezone
    now_london = datetime.now(pytz.timezone("Europe/London"))
    timestamp  = now_london.strftime("%A, %d %B %Y %H:%M %Z")
    header     = f"*Trading Signals — {timestamp}*\n\n"
    full_message = header + "\n".join(signals)

    print(full_message)
    send_telegram_message(full_message)

if __name__ == "__main__":
    main()
