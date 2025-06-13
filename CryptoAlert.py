import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
import numpy as np

from dotenv import load_dotenv
load_dotenv()

# --- CONFIG ---
KUCOIN_API_URL = "https://api.kucoin.com"
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
COINMARKETCAP_API_KEY = os.environ.get("COINMARKETCAP_API_KEY")
COINMARKETCAP_API_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
CRYPTOCOMPARE_API_KEY = os.environ.get("CRYPTOCOMPARE_API_KEY")
CRYPTOCOMPARE_NEWS_URL = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"

DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

CANDLE_INTERVAL = "30min"
WINDOW_DAYS = 7
RETRAIN_DAYS = 3

CONFIDENCE_THRESHOLD = 0.85
TP_PCT = 0.04
SL_PCT = -0.02
CANDLE_MIN = 30  # 30min candles
MAX_CANDLES = int(60 / (CANDLE_MIN / 60))  # 1 hour max duration (2 candles for 30min)
MIN_DURATION = 30
MAX_DURATION = 720

MIN_MARKET_CAP = 300_000_000
MIN_VOLUME = 1_000_000

def get_uk_time_header():
    now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
    uk_time = now_utc.astimezone(pytz.timezone("Europe/London"))
    return f"*🚀 CRYPTO TRADE SIGNAL  — {uk_time.strftime('%A, %d %B %Y %H:%M %Z')}*"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
        resp = requests.post(url, data=payload, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

def fetch_kucoin_usdt_symbols():
    try:
        resp = requests.get(f"{KUCOIN_API_URL}/api/v2/symbols", timeout=15)
        resp.raise_for_status()
        symbols = resp.json()['data']
        symbols_filtered = [
            {"symbol": s['symbol'], "base": s['baseCurrency'], "quote": s['quoteCurrency']}
            for s in symbols if s['symbol'].endswith('-USDT') and s['enableTrading']
        ]
        return symbols_filtered
    except Exception as e:
        return []

def fetch_coinmarketcap_market_data(min_market_cap=MIN_MARKET_CAP, min_volume=MIN_VOLUME, max_coins=2000):
    url = COINMARKETCAP_API_URL
    per_page = 500  # max per CMC API
    headers = {
        "X-CMC_PRO_API_KEY": COINMARKETCAP_API_KEY,
        "Accepts": "application/json"
    }
    filtered = {}
    for start in range(1, max_coins+1, per_page):
        params = {
            "start": start,
            "limit": per_page,
            "convert": "USD"
        }
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if not data:
                break
            for c in data:
                symbol = c["symbol"].upper()
                market_cap = c["quote"]["USD"]["market_cap"]
                volume_24h = c["quote"]["USD"]["volume_24h"]
                filtered[symbol] = {
                    "market_cap": market_cap,
                    "volume_24h": volume_24h,
                    "name": c.get("name"),
                }
            if len(data) < per_page:
                break
        except Exception:
            break
    return filtered

def fetch_kucoin_ohlcv(symbol, interval="30min", limit=200):
    limit = max(30, min(limit, 200))
    url = f"{KUCOIN_API_URL}/api/v1/market/candles"
    params = {"symbol": symbol, "type": interval, "limit": limit}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json().get('data', [])
        if not data or len(data) < 2:
            return None
        df = pd.DataFrame(data, columns=["time", "open", "close", "high", "low", "volume", "turnover"])
        df = df.iloc[::-1].reset_index(drop=True)  # oldest first
        df['time'] = pd.to_numeric(df['time'])
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        for col in ['open','close','high','low','volume','turnover']:
            df[col] = pd.to_numeric(df[col])
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        return df.reset_index(drop=True)
    except Exception:
        return None

def fetch_news_sentiment():
    headers = {'Authorization': f'Apikey {CRYPTOCOMPARE_API_KEY}'}
    coin_news_counts = {}
    try:
        resp = requests.get(CRYPTOCOMPARE_NEWS_URL, headers=headers, timeout=10)
        resp.raise_for_status()
        news_data = resp.json().get('Data', [])
        for article in news_data:
            try:
                tags = article.get('tags', [])
                title = article.get('title', '').lower()
                is_bullish = any(word in title for word in [
                    'rise', 'bull', 'breakout', 'gain', 'pump', 'spike', 'surge', 'rally'
                ])
                is_bearish = any(word in title for word in [
                    'fall', 'bear', 'drop', 'crash', 'dump', 'slump', 'plunge', 'collapse'
                ])
                for tag in tags:
                    if tag.endswith('USDT'):
                        key = tag.upper().replace("-", "")
                        if key not in coin_news_counts:
                            coin_news_counts[key] = {"bullish": 0, "bearish": 0}
                        if is_bullish:
                            coin_news_counts[key]["bullish"] += 1
                        if is_bearish:
                            coin_news_counts[key]["bearish"] += 1
            except Exception:
                pass
    except Exception:
        pass
    return coin_news_counts

# --- ATR-based TP/SL ML Duration Label ---
def compute_duration_to_tp_sl_atr(df, tp_mult=2, sl_mult=1, max_candles=MAX_CANDLES):
    durations = []
    for idx in range(len(df)):
        entry = df['close'].iloc[idx]
        atr = df['atr'].iloc[idx]
        tp = entry + tp_mult * atr
        sl = entry - sl_mult * atr
        duration = 0
        for forward in range(1, max_candles + 1):
            if idx + forward >= len(df):
                break
            high = df['high'].iloc[idx + forward]
            low = df['low'].iloc[idx + forward]
            if high >= tp:
                duration = forward
                break
            if low <= sl:
                duration = -forward
                break
        durations.append(duration)
    df['duration_to_tp_sl_atr'] = durations
    return df

# --- Legacy duration target for backward compatibility (relaxed ATR multiplier) ---
def compute_duration_target_atr(df, atr_mult=0.75, max_candles=MAX_CANDLES):
    targets = []
    for idx in range(len(df)):
        cur_price = df['close'].iloc[idx]
        cur_atr = df['atr'].iloc[idx]
        found = False
        for forward in range(1, max_candles+1):
            if idx + forward >= len(df):
                break
            future_price = df['close'].iloc[idx + forward]
            up_trigger = cur_price + atr_mult * cur_atr
            down_trigger = cur_price - atr_mult * cur_atr
            if future_price >= up_trigger or future_price <= down_trigger:
                targets.append(forward * CANDLE_MIN)
                found = True
                break
        if not found:
            targets.append(max_candles * CANDLE_MIN)
    df['duration_target'] = targets
    return df

def create_features_and_label(df):
    df['roc_5'] = df['close'].pct_change(5) * 100
    df['roc_15'] = df['close'].pct_change(15) * 100
    adx_indicator = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['adx'] = adx_indicator.adx()
    df['adx_pos'] = adx_indicator.adx_pos()
    df['adx_neg'] = adx_indicator.adx_neg()
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.dayofweek

    atr_indicator = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['atr'] = atr_indicator.average_true_range()

    # --- ATR-based TP/SL columns ---
    df['tp_atr'] = df['close'] + 2 * df['atr']
    df['sl_atr'] = df['close'] - 1 * df['atr']

    df['max_future_2'] = np.maximum(df['close'].shift(-1), df['close'].shift(-2))

    # --- Relaxed SPIKE LOGIC (0.75 x ATR) ---
    df['target'] = ((df['max_future_2'] - df['close']) > (0.75 * df['atr'])).astype(int)

    # --- ATR-based ML Duration Label ---
    df = compute_duration_to_tp_sl_atr(df, tp_mult=2, sl_mult=1, max_candles=MAX_CANDLES)
    # --- Legacy duration for reference ---
    df = compute_duration_target_atr(df, atr_mult=0.75, max_candles=MAX_CANDLES)
    return df.dropna()

def calc_dmi(df):
    adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    return adx.adx().iloc[-1], adx.adx_pos().iloc[-1], adx.adx_neg().iloc[-1]

def rule_based_check(df):
    price = df['close'].iloc[-1]
    # --- Relaxed breakout threshold (0.7 quantile) ---
    breakout_threshold = df['close'].iloc[-100:].quantile(0.7)
    roc_5 = df['close'].pct_change(5).iloc[-1] * 100
    adx_val, adx_pos, adx_neg = calc_dmi(df)
    bullish = price > breakout_threshold and roc_5 > 0.5 and adx_val > 20 and adx_pos > adx_neg
    return bullish, adx_val

def train_models(symbol):
    csv_path = os.path.join(DATA_DIR, f"{symbol.replace('-', '_')}.csv")
    if not os.path.exists(csv_path):
        return None, None
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = create_features_and_label(df)
    features = ['roc_5', 'roc_15', 'adx', 'adx_pos', 'adx_neg', 'hour', 'day', 'atr']
    X = df[features]
    y = df['target']
    y_reg = df['duration_to_tp_sl_atr'].clip(MIN_DURATION, MAX_DURATION)
    value_counts = y.value_counts()
    if len(value_counts) < 2 or (value_counts < 2).any():
        return None, None
    X_train, X_val, y_train, y_val, yreg_train, yreg_val = train_test_split(
        X, y, y_reg, test_size=0.2, stratify=y, random_state=42
    )
    model_cls = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
    model_cls.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    model_reg = xgb.XGBRegressor(objective='reg:squarederror')
    model_reg.fit(X_train, yreg_train, eval_set=[(X_val, yreg_val)], verbose=False)
    model_cls.save_model(os.path.join(MODEL_DIR, f"{symbol.replace('-', '_')}_xgb_cls.json"))
    model_reg.save_model(os.path.join(MODEL_DIR, f"{symbol.replace('-', '_')}_xgb_reg.json"))
    return model_cls, model_reg

def load_models(symbol):
    cls_path = os.path.join(MODEL_DIR, f"{symbol.replace('-', '_')}_xgb_cls.json")
    reg_path = os.path.join(MODEL_DIR, f"{symbol.replace('-', '_')}_xgb_reg.json")
    if not os.path.exists(cls_path) or not os.path.exists(reg_path):
        return None, None
    model_cls = xgb.XGBClassifier()
    model_cls.load_model(cls_path)
    model_reg = xgb.XGBRegressor()
    model_reg.load_model(reg_path)
    return model_cls, model_reg

def predict_breakout(df, model_cls):
    features = ['roc_5', 'roc_15', 'adx', 'adx_pos', 'adx_neg', 'hour', 'day', 'atr']
    latest = df.iloc[-1:]
    proba = model_cls.predict_proba(latest[features])[0][1]
    return proba

def predict_duration(df, model_reg):
    features = ['roc_5', 'roc_15', 'adx', 'adx_pos', 'adx_neg', 'hour', 'day', 'atr']
    latest = df.iloc[-1:]
    duration = model_reg.predict(latest[features])[0]
    duration = int(np.clip(round(duration), MIN_DURATION, MAX_DURATION))
    return duration

def save_ohlcv(symbol, df):
    path = os.path.join(DATA_DIR, f"{symbol.replace('-', '_')}.csv")
    df.to_csv(path, index=False)

def format_signal_line(base, symbol, confidence_pct, roc_5, roc_15, news_str, breakout_level, price, duration, sl, tp, sl_atr=None, tp_atr=None, duration_atr=None):
    msg = (
        f"🚀 *{base.upper()}* ({symbol.replace('-', '/')})\n"
        f"  Confidence: {confidence_pct}% | ROC5: {roc_5:.2f}% | ROC15: {roc_15:.2f}% | News: {news_str}\n"
        f"  Breakout > ${breakout_level:.4f} | Price: ${price:.4f} | ML-Duration: {duration}min\n"
        f"  SL: ${sl:.4f} | TP: ${tp:.4f}\n"
    )
    # --- ATR-based TP/SL/duration info ---
    if sl_atr is not None and tp_atr is not None:
        msg += f"  ATR-Stop: ${sl_atr:.4f} | ATR-Target: ${tp_atr:.4f}\n"
    if duration_atr is not None:
        if duration_atr > 0:
            msg += f"  ATR-ML duration: {duration_atr} bars (to TP)\n"
        elif duration_atr < 0:
            msg += f"  ATR-ML duration: {-duration_atr} bars (to SL)\n"
        else:
            msg += f"  ATR-ML duration: not hit in 1h\n"
    msg += f"  Max duration: 1 hour\n"
    return msg

def main():
    kucoin_symbols = fetch_kucoin_usdt_symbols()
    kucoin_bases = set([sym['base'].upper() for sym in kucoin_symbols])
    cmc_data = fetch_coinmarketcap_market_data()
    available_bases = []
    for base in kucoin_bases:
        cmc_entry = cmc_data.get(base)
        if cmc_entry:
            if (cmc_entry["market_cap"] is not None and cmc_entry["market_cap"] >= MIN_MARKET_CAP and
                cmc_entry["volume_24h"] is not None and cmc_entry["volume_24h"] >= MIN_VOLUME):
                available_bases.append(base)

    news_sentiment = fetch_news_sentiment()
    market_bullish_news = 0
    market_bearish_news = 0
    try:
        resp = requests.get(CRYPTOCOMPARE_NEWS_URL, headers={'Authorization': f'Apikey {CRYPTOCOMPARE_API_KEY}'}, timeout=10)
        resp.raise_for_status()
        news_data = resp.json().get('Data', [])
        for article in news_data:
            title = article.get('title', '').lower()
            is_bullish = any(word in title for word in [
                'rise', 'bull', 'breakout', 'gain', 'pump', 'spike', 'surge', 'rally',
                'altcoin season', 'altcoin rally', 'crypto surge', 'market breakout', 'ath', 'all time high'
            ])
            is_bearish = any(word in title for word in [
                'fall', 'bear', 'drop', 'crash', 'dump', 'slump', 'plunge', 'collapse', 'market correction'
            ])
            if is_bullish:
                market_bullish_news += 1
            if is_bearish:
                market_bearish_news += 1
    except Exception:
        pass

    analyzed_pairs = []
    skipped_pairs = []
    signals_sent = 0
    bullish_news_total = 0
    bearish_news_total = 0

    signals = []
    for sym_info in kucoin_symbols:
        symbol = sym_info['symbol']
        base = sym_info['base'].upper()
        quote = sym_info['quote']
        if base not in available_bases:
            skipped_pairs.append(symbol)
            continue

        df = fetch_kucoin_ohlcv(symbol, interval=CANDLE_INTERVAL, limit=200)
        if df is None:
            skipped_pairs.append(symbol)
            continue
        if len(df) < 30:
            skipped_pairs.append(symbol)
            continue

        save_ohlcv(symbol, df)

        model_cls, model_reg = load_models(symbol)
        cls_path = os.path.join(MODEL_DIR, f"{symbol.replace('-', '_')}_xgb_cls.json")
        reg_path = os.path.join(MODEL_DIR, f"{symbol.replace('-', '_')}_xgb_reg.json")
        retrain_needed = True
        if os.path.exists(cls_path) and os.path.exists(reg_path):
            mtime = datetime.utcfromtimestamp(os.path.getmtime(cls_path))
            if (datetime.utcnow() - mtime).days < RETRAIN_DAYS:
                retrain_needed = False
        if model_cls is None or model_reg is None or retrain_needed:
            model_cls, model_reg = train_models(symbol)
            if model_cls is None or model_reg is None:
                skipped_pairs.append(symbol)
                continue

        df = create_features_and_label(df)
        bullish, adx_val = rule_based_check(df)
        if not bullish:
            skipped_pairs.append(symbol)
            continue

        proba = predict_breakout(df, model_cls)

        coin_key = symbol.replace("-", "").upper()
        news_counts = news_sentiment.get(coin_key, {"bullish": 0, "bearish": 0})
        bullish_news = news_counts["bullish"]
        bearish_news = news_counts["bearish"]
        bullish_news_total += bullish_news
        bearish_news_total += bearish_news

        total_bullish = bullish_news + market_bullish_news // 5
        total_bearish = bearish_news + market_bearish_news // 5

        if total_bullish + total_bearish > 0:
            news_factor = 0.5 + 0.25 * (total_bullish - total_bearish) / (total_bullish + total_bearish)
        else:
            news_factor = 0.5

        confidence = proba * 0.8 + news_factor * 0.2
        if confidence < CONFIDENCE_THRESHOLD:
            analyzed_pairs.append(symbol)
            continue

        roc_5 = df['roc_5'].iloc[-1] if 'roc_5' in df.columns else df['close'].pct_change(5).iloc[-1] * 100
        roc_15 = df['roc_15'].iloc[-1] if 'roc_15' in df.columns else df['close'].pct_change(15).iloc[-1] * 100
        breakout_level = df['close'].iloc[-100:].quantile(0.7)
        price = df['close'].iloc[-1]
        duration = predict_duration(df, model_reg)
        sl = price * 0.985
        tp = price * 1.04
        sl_atr = df['sl_atr'].iloc[-1]
        tp_atr = df['tp_atr'].iloc[-1]
        duration_atr = df['duration_to_tp_sl_atr'].iloc[-1] if 'duration_to_tp_sl_atr' in df.columns else None
        confidence_pct = int(confidence * 100)
        news_str = f"B{total_bullish}/R{total_bearish}"

        signals.append(format_signal_line(
            base, symbol, confidence_pct, roc_5, roc_15, news_str,
            breakout_level, price, duration, sl, tp, sl_atr, tp_atr, duration_atr
        ))
        signals_sent += 1
        analyzed_pairs.append(symbol)

    print("\n===== FINAL SUMMARY =====")
    print(f"Analyzed pairs: {len(analyzed_pairs)}")
    print(f"Skipped pairs: {len(skipped_pairs)}")
    print(f"Sent signals: {signals_sent}")
    print(f"Bullish news items: {bullish_news_total} (market-wide: {market_bullish_news})")
    print(f"Bearish news items: {bearish_news_total} (market-wide: {market_bearish_news})")
    print(f"--- {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC - Bot Finished ---")

    if signals:
        uk_time = get_uk_time_header()
        full_message = (
            f"{uk_time}\n\n"
            f"*ALERT: {signals_sent} Trade Signals*\n\n" +
            "\n".join(signals)
        )
        send_telegram_message(full_message)
    else:
        send_telegram_message(f"{get_uk_time_header()}\n\nNo qualifying trade signals found this run.")

if __name__ == "__main__":
    main()
