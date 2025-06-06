import os
from datetime import datetime
import pytz
import requests

# Load Telegram credentials from environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Assets and message parameters
ASSETS = ["EUR/USD", "GBP/USD", "XAU/USD"]
TIME = "08:00-22:00 BST"
FOCUS_Time = "Focus on 13:00–16:00 BST"
REASON = "London & New York overlap"

def send_telegram_message(text: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    try:
        resp = requests.post(url, data=payload, timeout=10)
        return resp.status_code == 200
    except Exception:
        return False

def should_send() -> bool:
    """
    Returns True if current UK time is exactly 08:00 on a weekday (Mon–Fri).
    """
    london_now = datetime.now(pytz.timezone("Europe/London"))
    is_weekday = london_now.weekday() < 5        # 0 = Monday, 4 = Friday
    current_time = london_now.strftime("%H:%M")  # e.g., "08:00"
    return is_weekday and current_time == START_TIME

def build_daily_message() -> str:
    """
    Constructs the Telegram message for the day.
    """
    london_now = datetime.now(pytz.timezone("Europe/London"))
    today_str = london_now.strftime("%A, %d %B %Y")
    assets_line = ", ".join(ASSETS)
    return (
        f"*Daily Trading Alert — {today_str}*\n\n"
        f"*Assets*: {assets_line}\n"
        f"*Start Time*: {START_TIME} BST\n"
        f"*{FOCUS_NOTE}*\n"
        f"*Reason*: {REASON}\n\n"
        "🔔 Ready to trade? Stay disciplined and honour your stops!"
    )

if __name__ == "__main__":
    if should_send():
        message = build_daily_message()
        success = send_telegram_message(message)
        if not success:
            print("Failed to send Telegram message.")