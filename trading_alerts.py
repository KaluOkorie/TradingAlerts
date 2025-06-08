import os
import ccxt
import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz

# ─── CONFIG ────────────────────────────────────────────────────────────────────
# Make sure to set these environment variables in your system
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

VOLUME_THRESH      = 1_000_000       # $1M 24h volume
MARKETCAP_THRESH   = 50_000_000      # $50M market cap
TIMEFRAME          = '4h'
PROFIT_GOAL_USD    = 5.00            # ## NEW ## Minimum desired profit in USD per trade

# ─── UTILS ─────────────────────────────────────────────────────────────────────
exchange = ccxt.kraken({ 'enableRateLimit': True })

def send_telegram(text: str):
    """Sends a message to a Telegram chat."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = { 'chat_id': TELEGRAM_CHAT_ID, 'text': text, 'parse_mode': 'Markdown' }
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Error sending Telegram message: {e}")

def get_uk_time_header():
    """Gets the current UK time and formats it for the message header."""
    now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
    uk_time = now_utc.astimezone(pytz.timezone("Europe/London"))
    # ## NEW ## Updated format to match the user request
    return f"📡 *Crypto Alerts — {uk_time.strftime('%A, %d %B %Y %H:%M %Z')}*"

# ─── 1) FILTER PAIRS BY VOLUME & MARKETCAP ────────────────────────────────────
def get_candidate_symbols():
    """Filters trading pairs by 24h volume and market capitalization."""
    print("Filtering pairs by volume and market cap...")
    exchange.load_markets()
    # Fetch tickers for pairs ending in /USD or /USDT
    tickers = exchange.fetch_tickers([s for s in exchange.symbols if s.endswith('/USD') or s.endswith('/USDT')])
    high_vol = { sym for sym, data in tickers.items() if data.get('quoteVolume', 0) >= VOLUME_THRESH }

    # Fetch market cap data from CoinGecko
    cg_url = "https://api.coingecko.com/api/v3/coins/markets"
    cg_params = { 'vs_currency': 'usd', 'order':'market_cap_desc', 'per_page':250, 'page':1 }
    cg_response = requests.get(cg_url, params=cg_params)
    cg = cg_response.json()

    # Create a set of symbols that meet the market cap threshold
    market_cap_ok = { f"{item['symbol'].upper()}/USD" for item in cg if item.get('market_cap', 0) >= MARKETCAP_THRESH }

    # Find the intersection of both sets and sort them
    candidates = sorted(list(high_vol & market_cap_ok))
    print(f"Found {len(candidates)} candidates.")
    return candidates

# ─── 2) FETCH & SIGNAL TEST ────────────────────────────────────────────────────
def fetch_ohlcv(symbol: str):
    """Fetches OHLCV data for a given symbol."""
    since = exchange.parse8601((datetime.utcnow() - timedelta(days=30)).isoformat()) # Fetch more data for ATR stability
    ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, since)
    df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','vol'])
    df.set_index(pd.to_datetime(df['ts'], unit='ms'), inplace=True)
    return df[['open','high','low','close','vol']]

def compute_signal(df: pd.DataFrame, sym: str):
    """Computes technical indicators and generates a trade signal if conditions are met."""
    if len(df) < 30: return None

    # ## NEW ## Calculate ATR (Average True Range)
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['ATR'] = df['TR'].ewm(span=14, adjust=False).mean()

    # Standard Indicators
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

    # ─── Signal Conditions ─────────────────────────────────────────────────────
    is_bullish = all([
        last.EMA9  > last.EMA21,
        last.RSI   > 50,
        last.MACD  > last.Signal,
        last.close > last.open,
        breakout_up,
        retest_up
    ])

    is_bearish = all([
        last.EMA9  < last.EMA21,
        last.RSI   < 50,
        last.MACD  < last.Signal,
        last.close < last.open,
        breakout_down,
        retest_down
    ])

    if is_bullish:
        # ## NEW ## Dynamic TP/SL and message formatting for bullish signal
        entry_price = last.close
        last_atr = last.ATR

        # SL cannot be zero or negative
        if last_atr <= 0: return None

        take_profit = entry_price + (0.5 * last_atr)
        stop_loss = entry_price - (0.2 * last_atr)

        # Ensure the trade meets the minimum profit goal
        if (take_profit - entry_price) < PROFIT_GOAL_USD:
            return None

        risk_pct = (abs(entry_price - stop_loss) / entry_price) * 100
        reward_pct = (abs(take_profit - entry_price) / entry_price) * 100

        return (
            f"🔔 *{sym}* — Consider 📈*BUY*\n"
            f"📍 Breakout + Retest of Resistance at *${recent_high:,.2f}* → _Current Price_: *${entry_price:,.2f}*\n"
            f"🟩 Entry:  `${entry_price:,.2f}`\n"
            f"🎯 TP: `${take_profit:,.2f}` | 🛑 SL: `${stop_loss:,.2f}`\n"
            f"📊 ATR(14): `${last_atr:,.2f}` | Risk: {risk_pct:.1f}%, Reward: {reward_pct:.1f}% | Direction: *Bullish*"
        )

    if is_bearish:
        # ## NEW ## Dynamic TP/SL and message formatting for bearish signal
        entry_price = last.close
        last_atr = last.ATR

        # SL cannot be zero or negative
        if last_atr <= 0: return None

        # Note: Formulas are reversed for a short/sell position
        take_profit = entry_price - (0.5 * last_atr)
        stop_loss = entry_price + (0.2 * last_atr)

        # Ensure the trade meets the minimum profit goal
        if (entry_price - take_profit) < PROFIT_GOAL_USD:
            return None

        risk_pct = (abs(entry_price - stop_loss) / entry_price) * 100
        reward_pct = (abs(take_profit - entry_price) / entry_price) * 100

        return (
            f"🔔 *{sym}* — Consider 📉*SELL*\n"
            f"📍 Breakdown + Retest of Support at *${recent_low:,.2f}* → _Current Price_: *${entry_price:,.2f}*\n"
            f"🟥 Entry:  `${entry_price:,.2f}`\n"
            f"🎯 TP: `${take_profit:,.2f}` | 🛑 SL: `${stop_loss:,.2f}`\n"
            f"📊 ATR(14): `${last_atr:,.2f}` | Risk: {risk_pct:.1f}%, Reward: {reward_pct:.1f}% | Direction: *Bearish*"
        )

    return None

# ─── 3) RUN & NOTIFY ───────────────────────────────────────────────────────────
def main():
    """Main function to run the alert scanner."""
    if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
        print("Error: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables must be set.")
        return

    candidates = get_candidate_symbols()
    alerts = []
    print(f"\nScanning {len(candidates)} candidates for signals...")

    for i, sym in enumerate(candidates):
        print(f"  [{i+1}/{len(candidates)}] Checking {sym}...")
        try:
            df = fetch_ohlcv(sym)
            result = compute_signal(df, sym)
            if result:
                print(f"    -> Signal FOUND for {sym}!")
                alerts.append(result)
        except ccxt.NetworkError as e:
            print(f"    -> Network error fetching {sym}: {e}")
            continue
        except Exception as e:
            print(f"    -> An unexpected error occurred with {sym}: {e}")
            continue

    if alerts:
        print(f"\nFound {len(alerts)} alerts. Sending to Telegram.")
        header = get_uk_time_header()
        message = header + "\n\n" + "\n\n".join(alerts)
        send_telegram(message)
    else:
        print("\nScan complete. No alerts to send.")

if __name__ == "__main__":
    main()
