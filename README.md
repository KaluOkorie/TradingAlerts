# Automated Trading Signal Bot (1-Hour Interval)
This Python-based trading signal bot fetches real-time financial data from Yahoo Finance using yfinance, 
analyzes assets across various markets, and sends actionable long/short signals via Telegram every hour.

# Assets Monitored
Forex Pairs
EUR/USD (EURUSD=X)

GBP/USD (GBPUSD=X)

USD/JPY (USDJPY=X)

AUD/USD (AUDUSD=X)

NZD/USD (NZDUSD=X)

USD/CAD (USDCAD=X)

USD/CHF (USDCHF=X)

EUR/JPY (EURJPY=X)

EUR/GBP (EURGBP=X)

# Global Indices
S&P 500 (^GSPC)

NASDAQ Composite (^IXIC)

Dow Jones (^DJI)

DAX (Germany) (^GDAXI)

FTSE 100 (UK) (^FTSE)

Euro Stoxx 50 (^STOXX50E)

Nikkei 225 (Japan) (^N225)

Hang Seng (HK) (^HSI)

ASX 200 (Australia) (^AXJO)

Bovespa (Brazil) (^BVSP)

# Commodities / Metals / Energy
Gold Futures (GC=F)

Silver Futures (SI=F)

Crude Oil (WTI) (CL=F)

Natural Gas (NG=F)

Corn (ZC=F)

Soybean (ZS=F)

Copper (HG=F)

Platinum (PL=F)

# Popular ETFs
SPDR S&P 500 ETF (SPY)

Invesco QQQ Trust (QQQ)

iShares Russell 2000 ETF (IWM)

Vanguard Total Market ETF (VTI)

Vanguard Developed Markets ETF (VEA)

iShares US Aggregate Bond ETF (AGG)

Financial Sector ETF (XLF)

Health Care Sector ETF (XLV)

# Blue-Chip Stocks
Apple (AAPL)

Microsoft (MSFT)

Alphabet / Google (GOOGL)

Amazon (AMZN)

Tesla (TSLA)

Johnson & Johnson (JNJ)

Coca-Cola (KO)

Walmart (WMT)

Visa (V)

Procter & Gamble (PG)

NVIDIA (NVDA)

Meta (Facebook) (META)

# Crypto Alert Bot

The Python script (CryptoAlert.py) also monitor top USD/USDT 
cryptocurrency pairs and send concise buy or sell signals to the 
Telegram bot based on a 5‑condition confluence strategy, Signals are evaluated and sent every 4 hours.

# 🧠 Core Logic
The bot performs 5 major computations for each asset every hour:

**Relative Strength Index (RSI)** — Detects overbought/oversold conditions.

**MACD + Signal Line** — Identifies trend strength and direction.

**EMA Crossovers** — Short-term vs long-term moving average alignment.

**Volume Spike Detection** — Filters trades based on abnormal trading volume.

**Volatility Check (ATR or std dev)** — Ensures signals occur in tradable conditions.

# 🛠️ Requirements
Python 3.9+

yfinance

pandas

ta (technical analysis)

python-telegram-bot

dotenv

# 📬 Telegram Notifications
Time Interval: Every 1 hour.

If a bullish (long) or bearish (short) trading signal is detected for any asset, the bot sends a formatted message to Telegram.

If no actionable signal is detected, the bot remains silent (no spam).


