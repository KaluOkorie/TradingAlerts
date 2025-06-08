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
ASSETS = {
    # ─── Forex Pairs ───────────────────────────────
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "USDJPY=X": "USD/JPY",
    "AUDUSD=X": "AUD/USD",
    "NZDUSD=X": "NZD/USD",
    "USDCAD=X": "USD/CAD",

    # ─── Indices ───────────────────────────────────
    "^GSPC":      "S&P 500",
    "^IXIC":      "NASDAQ Composite",
    "^DJI":       "Dow Jones Industrial",
    "^GDAXI":     "DAX (Germany)",
    "^FTSE":      "FTSE 100 (UK)",
    "^STOXX50E":  "Euro Stoxx 50",
    "^N225":      "Nikkei 225",
    "^HSI":       "Hang Seng Index",

    # ─── Commodities / Futures ─────────────────────
    "GC=F":  "Gold Futures",
    "SI=F":  "Silver Futures",
    "CL=F":  "Crude Oil (WTI)",
    "NG=F":  "Natural Gas",
    "ZC=F":  "Corn",
    "ZS=F":  "Soybeans",
    "HG=F":  "Copper",
    "PL=F":  "Platinum",

    # ─── Cryptocurrencies ──────────────────────────
    "BTC-USD":    "Bitcoin",
    "ETH-USD":    "Ethereum",
    "SOL-USD":    "Solana",
    "BNB-USD":    "Binance Coin",
    "ADA-USD":    "Cardano",
    "XRP-USD":    "Ripple",
    "DOGE-USD":  "Dogecoin",
    "AVAX-USD":  "Avalanche",
    "DOT-USD":    "Polkadot",
    "USDT-USD":  "Tether (USDT)",
    "USDC-USD":  "USD Coin",

    # ─── Stocks ────────────────────────────────────
    "AAPL":    "Apple",
    "MSFT":    "Microsoft",
    "GOOGL":  "Alphabet (Google)",
    "META":    "Meta Platforms (Facebook)",
    "AMZN":    "Amazon",
    "NVDA":    "NVIDIA",
    "TSLA":    "Tesla",
    "JPM":    "JPMorgan Chase",
    "V":      "Visa",
    "MA":      "Mastercard",
    "WMT":    "Walmart",
    "PG":      "Procter & Gamble",
    "KO":      "Coca-Cola",
    "JNJ":    "Johnson & Johnson",
    "UNH":    "UnitedHealth Group",
    "PEP":    "PepsiCo",
    "HD":      "Home Depot",
    "CVX":    "Chevron",
    "XOM":    "ExxonMobil",
    "BABA":    "Alibaba",
    "NFLX":    "Netflix",
    "INTC":    "Intel",

    # ─── Equity ETFs ───────────────────────────────
    "SPY":  "SPDR S&P 500 ETF",
    "QQQ":  "Invesco QQQ Trust",
    "DIA":  "SPDR Dow Jones ETF",
    "IWM":  "iShares Russell 2000 ETF",
    "VOO":  "Vanguard S&P 500 ETF",
    "VTI":  "Vanguard Total Stock Market ETF",
    "VEA":  "Vanguard Developed Markets ETF",
    "VWO":  "Vanguard Emerging Markets ETF",

    # ─── Sector ETFs ───────────────────────────────
    "XLK":  "Technology Select Sector SPDR",
    "XLF":  "Financial Select Sector SPDR",
    "XLV":  "Health Care Select Sector SPDR",
    "XLE":  "Energy Select Sector SPDR",
    "XLY":  "Consumer Discretionary SPDR",
    "XLP":  "Consumer Staples Select Sector SPDR",
    "XLI":  "Industrial Select Sector SPDR",
    "XLU":  "Utilities Select Sector SPDR",

    # ─── Commodity ETFs ────────────────────────────
    "GLD":  "SPDR Gold Shares",
    "SLV":  "iShares Silver Trust",
    "USO":  "United States Oil Fund",
    "DBA":  "Invesco Agriculture Fund",
    "PPLT": "Aberdeen Physical Platinum Shares ETF",

    # ─── Bond & Treasury ETFs ──────────────────────
    "BND":    "Vanguard Total Bond Market ETF",
    "AGG":    "iShares Core US Aggregate Bond ETF",
    "TLT":    "iShares 20+ Year Treasury Bond ETF",
    "IEF":    "iShares 7-10 Year Treasury Bond ETF",
    "TIP":    "iShares TIPS Bond ETF"
}

SHORT_EMA  = 9
LONG_EMA   = 21
RSI_PERIOD = 14

def send_telegram_message(text: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       text,
        "parse_mode": "Markdown",
    }
    try:
        resp = requests.post(url, data=payload, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print("Telegram send error:", e)
        return False

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure enough data points for all indicators
    min_len_for_indicators = max(SHORT_EMA, LONG_EMA, RSI_PERIOD, 26) # 26 for MACD's longest EMA
    if len(df) < min_len_for_indicators:
        return pd.DataFrame() # Return an empty DataFrame if not enough data

    df['EMA9']  = df['Close'].ewm(span=SHORT_EMA, adjust=False).mean()
    df['EMA21'] = df['Close'].ewm(span=LONG_EMA, adjust=False).mean()
    
    delta     = df['Close'].diff()
    gain      = delta.where(delta > 0, 0.0)
    loss      = -delta.where(delta < 0, 0.0)
    
    # Using ewm for RSI, similar to the other code, as it's more common for real-time calculation
    # Handle potential division by zero for RSI
    avg_gain = gain.ewm(span=RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(span=RSI_PERIOD, adjust=False).mean()
    
    # Prevent division by zero if avg_loss is 0
    # Using 1e-10 is a common technique to avoid dividing by absolute zero
    rs = avg_gain / avg_loss.replace(0, 1e-10) 
    
    df['RSI'] = 100 - (100 / (1 + rs))
    # Fill NaN values that might occur at the beginning of the RSI series
    df['RSI'] = df['RSI'].fillna(50) 

    df['MACD']   = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df.dropna(subset=['EMA9','EMA21','RSI','MACD','Signal']) # Drop rows where indicators are NaN

def analyze_asset(symbol: str, name: str) -> str | None:
    # Use a longer period for data download to ensure sufficient data points
    # 7 days with 60m interval might be too short for 26-period EMA or 14-period RSI
    # Consider "30d" or "60d" for robustness. Let's use 30d here.
    df = yf.download(symbol, interval="60m", period="30d", auto_adjust=True, progress=False)
    
    if df.empty:
        print(f"    No data downloaded for {name} ({symbol}). Skipping.")
        return None
    
    df = calculate_indicators(df)
    
    if df.empty or len(df) < 2: # Need at least two rows for current and previous values
        print(f"    Insufficient data after indicator calculation for {name} ({symbol}). Skipping.")
        return None
        
    last, prev = df.iloc[-1], df.iloc[-2]

    # --- Crucial Fix: Remove bool() casts ---
    # The comparison itself results in a boolean (True/False), no explicit casting needed
    # Also, ensure no NaN values are involved in the comparisons
    if any(pd.isna([last['EMA9'], last['EMA21'], last['RSI'], last['MACD'], last['Signal'], last['Close'], last['Open'], prev['Close']])):
        print(f"    One or more essential indicator/price values are NaN for {name} ({symbol}). Skipping signal check.")
        return None

    bullish = all([
        last['EMA9']  > last['EMA21'],
        last['RSI']   > 50,
        last['MACD']  > last['Signal'],
        last['Close'] > prev['Close'],
        last['Close'] > last['Open']
    ])
    
    bearish = all([
        last['EMA9']  < last['EMA21'],
        last['RSI']   < 50,
        last['MACD']  < last['Signal'],
        last['Close'] < prev['Close'],
        last['Close'] < last['Open']
    ])
    
    if bullish:
        return f"🟢 *{name} ({symbol})*: Consider going *LONG* — All 5 bullish conditions met."
    if bearish:
        return f"🔴 *{name} ({symbol})*: Consider going *SHORT* — All 5 bearish conditions met."
        
    return None

def main():
    # Only Mon–Fri 08:00–23:00 London time
    now = datetime.now(pytz.timezone("Europe/London"))
    
    # Check if the market is open during scan time
    # This logic assumes most markets relevant to these assets are open during these hours.
    # For a more robust check, you might need specific market hours for each asset type.
    if now.weekday() > 4 or not (8 <= now.hour < 23):
        print(f"Current time ({now.strftime('%H:%M %Z')}) is outside trading hours (Mon-Fri, 08:00-23:00 London time). Exiting.")
        return

    print(f"Scanner initiated. Analyzing {len(ASSETS)} assets...")
    signals = []
    for sym, nm in ASSETS.items():
        try:
            print(f"  -> Scanning {nm} ({sym})...")
            msg = analyze_asset(sym, nm)
            if msg:
                print(f"  ✅ Signal found for {nm}!")
                signals.append(msg)
        except Exception as e:
            # This will catch any unexpected errors during analysis of a single asset
            print(f"  ❌ Error analyzing {nm} ({sym}): {e}")
            continue

    if signals:
        header = "*Trading Signals — " + now.strftime("%A, %d %B %Y %H:%M %Z") + "*\n\n"
        full = header + "\n".join(signals)
        print("\n--- SENDING TELEGRAM ALERT ---")
        print(full)
        send_telegram_message(full)
    else:
        print("\nScan complete. No high-quality signals found.")

if __name__ == "__main__":
    main()
