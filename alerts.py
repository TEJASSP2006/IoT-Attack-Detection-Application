import os
import smtplib
import ssl
import urllib.parse
import urllib.request
from email.mime.text import MIMEText


def send_telegram_alert(message: str) -> None:
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not bot_token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = urllib.parse.urlencode({"chat_id": chat_id, "text": message}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    with urllib.request.urlopen(req, timeout=8):
        pass


def send_email_alert(message: str) -> None:
    host = os.getenv("ALERT_SMTP_HOST", "").strip()
    port = int(os.getenv("ALERT_SMTP_PORT", "587"))
    username = os.getenv("ALERT_SMTP_USER", "").strip()
    password = os.getenv("ALERT_SMTP_PASSWORD", "").strip()
    sender = os.getenv("ALERT_EMAIL_FROM", "").strip()
    receiver = os.getenv("ALERT_EMAIL_TO", "").strip()
    if not all([host, username, password, sender, receiver]):
        return

    msg = MIMEText(message)
    msg["Subject"] = "IoT Attack Alert"
    msg["From"] = sender
    msg["To"] = receiver

    context = ssl.create_default_context()
    with smtplib.SMTP(host, port, timeout=10) as server:
        server.starttls(context=context)
        server.login(username, password)
        server.sendmail(sender, [receiver], msg.as_string())


def dispatch_alert(message: str) -> None:
    try:
        send_telegram_alert(message)
    except Exception:
        pass
    try:
        send_email_alert(message)
    except Exception:
        pass
