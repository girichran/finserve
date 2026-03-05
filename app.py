from flask import Flask, render_template, request, jsonify, g, session, redirect
import os
import random
import json
import smtplib
import ssl
import secrets
import hmac
import hashlib
import base64
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from email.message import EmailMessage

import psycopg2
from psycopg2.extras import RealDictCursor
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv

load_dotenv()

try:
    import certifi
except ImportError:
    certifi = None

try:
    import yfinance as yf
except ImportError:
    yf = None

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", os.urandom(32).hex())
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "2004")
DB_NAME = os.getenv("DB_NAME", "finserve")
DATABASE_URL = os.getenv("DATABASE_URL", "")
EMAIL_PROVIDER = os.getenv("EMAIL_PROVIDER", "smtp").strip().lower()
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")
CONTACT_TO_EMAIL = os.getenv("CONTACT_TO_EMAIL", SMTP_USER)
CONTACT_FROM_EMAIL = os.getenv("CONTACT_FROM_EMAIL", SMTP_USER)
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")
RAZORPAY_API_BASE = os.getenv("RAZORPAY_API_BASE", "https://api.razorpay.com")
ADMIN_EMAILS = {
    email.strip().lower()
    for email in os.getenv("ADMIN_EMAILS", "giricharan4321@gmail.com").split(",")
    if email.strip()
}
ENV = os.getenv("FLASK_ENV", "production").lower()
SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "true").lower() == "true"
SESSION_COOKIE_HTTPONLY = os.getenv("SESSION_COOKIE_HTTPONLY", "true").lower() == "true"
SESSION_COOKIE_SAMESITE = os.getenv("SESSION_COOKIE_SAMESITE", "Lax")
INDEX_SYMBOLS = [
    ("NIFTY 50", "^NSEI"),
    ("BANK NIFTY", "^NSEBANK"),
    ("NIFTY IT", "^CNXIT"),
    ("NIFTY AUTO", "^CNXAUTO"),
    ("NIFTY FMCG", "^CNXFMCG"),
    ("NIFTY PHARMA", "^CNXPHARMA"),
    ("NIFTY METAL", "^CNXMETAL"),
]
MARKET_HEATMAP_UNIVERSE = {
    "IT": [
        ("Infosys", "INFY.NS"),
        ("TCS", "TCS.NS"),
        ("Wipro", "WIPRO.NS"),
    ],
    "Banking": [
        ("HDFC Bank", "HDFCBANK.NS"),
        ("ICICI Bank", "ICICIBANK.NS"),
        ("SBI", "SBIN.NS"),
    ],
    "Energy": [
        ("Reliance", "RELIANCE.NS"),
        ("ONGC", "ONGC.NS"),
        ("BPCL", "BPCL.NS"),
    ],
    "Auto": [
        ("Tata Motors", "TATAMOTORS.NS"),
        ("Maruti", "MARUTI.NS"),
        ("M&M", "M&M.NS"),
    ],
}
DEFAULT_FED_EVENTS = [
    {"title": "FOMC Meeting (Estimated Window)", "date": "2026-03-18", "type": "Fed"},
    {"title": "FOMC Meeting (Estimated Window)", "date": "2026-05-06", "type": "Fed"},
    {"title": "FOMC Meeting (Estimated Window)", "date": "2026-06-17", "type": "Fed"},
]
SUBSCRIPTION_PLAN_CATALOG = {
    "STARTER_MONTHLY": {"name": "Starter (Retail)", "price_inr": 499.0, "duration_days": 30},
    "PRO_MONTHLY": {"name": "Pro Trader", "price_inr": 1499.0, "duration_days": 30},
    "ELITE_MONTHLY": {"name": "Elite Advisory", "price_inr": 4999.0, "duration_days": 30},
}

app.config.update(
    SESSION_COOKIE_SECURE=SESSION_COOKIE_SECURE,
    SESSION_COOKIE_HTTPONLY=SESSION_COOKIE_HTTPONLY,
    SESSION_COOKIE_SAMESITE=SESSION_COOKIE_SAMESITE,
    PERMANENT_SESSION_LIFETIME=60 * 60 * 12,
)


def _send_email(to_email, subject, text_body, reply_to=None):
    if not to_email:
        return False, "Recipient email is missing"

    provider = EMAIL_PROVIDER or "smtp"
    from_email = CONTACT_FROM_EMAIL or SMTP_USER

    if provider == "sendgrid":
        if not SENDGRID_API_KEY:
            return False, "SendGrid is not configured"
        if not from_email:
            return False, "CONTACT_FROM_EMAIL is not configured"

        payload = {
            "personalizations": [{"to": [{"email": to_email}]}],
            "from": {"email": from_email},
            "subject": subject,
            "content": [{"type": "text/plain", "value": text_body}],
        }
        if reply_to:
            payload["reply_to"] = {"email": reply_to}

        req = Request(
            "https://api.sendgrid.com/v3/mail/send",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {SENDGRID_API_KEY}",
            },
            method="POST",
        )
        try:
            ssl_context = ssl.create_default_context(cafile=certifi.where()) if certifi else ssl.create_default_context()
            with urlopen(req, timeout=20, context=ssl_context) as response:
                status = getattr(response, "status", 200)
                if int(status) in {200, 202}:
                    return True, "Email sent"
                return False, f"SendGrid failed with status {status}"
        except Exception as exc:
            return False, str(exc)

    if not SMTP_HOST or not SMTP_USER or not SMTP_PASSWORD:
        return False, "SMTP is not configured"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email
    if reply_to:
        msg["Reply-To"] = reply_to
    msg.set_content(text_body)

    try:
        if SMTP_PORT == 465:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=20) as server:
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(msg)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(msg)
        return True, "Email sent"
    except Exception as exc:
        return False, str(exc)


def send_contact_email(name, email, message):
    if not CONTACT_TO_EMAIL:
        return False, "CONTACT_TO_EMAIL is not configured"
    body = f"New contact message received.\n\nName: {name}\nEmail: {email}\n\nMessage:\n{message}\n"
    return _send_email(CONTACT_TO_EMAIL, f"New Contact Message from {name}", body, reply_to=email)


def send_password_reset_otp_email(email, otp_code):
    body = (
        f"Your OTP for password reset is: {otp_code}\n"
        f"This OTP is valid for 10 minutes.\n"
        f"If you did not request this, ignore this email."
    )
    return _send_email(email, "Your FinServe Password Reset OTP", body)


def send_stop_loss_alert_email(to_email, symbol, stop_loss, current_price, trigger_time):
    body = (
        "Stop Loss Trigger Alert\n\n"
        f"Symbol: {symbol}\n"
        f"Configured Stop Loss: {stop_loss}\n"
        f"Triggered Price: {current_price}\n"
        f"Trigger Time: {trigger_time}\n\n"
        "Trigger details: Price touched or moved below your stop loss level.\n"
        "Please review your position immediately."
    )
    return _send_email(to_email, f"Stop Loss Triggered: {symbol}", body)


def create_razorpay_order(amount_paise, receipt, notes=None):
    if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
        return None, "Razorpay is not configured"

    payload = {
        "amount": int(amount_paise),
        "currency": "INR",
        "receipt": receipt,
        "payment_capture": 1,
        "notes": notes or {},
    }

    credentials = f"{RAZORPAY_KEY_ID}:{RAZORPAY_KEY_SECRET}".encode("utf-8")
    auth_token = base64.b64encode(credentials).decode("utf-8")
    req = Request(
        f"{RAZORPAY_API_BASE.rstrip('/')}/v1/orders",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Basic {auth_token}",
        },
        method="POST",
    )

    try:
        with urlopen(req, timeout=15) as response:
            body = response.read().decode("utf-8")
            return json.loads(body), None
    except Exception as exc:
        return None, str(exc)


def verify_razorpay_signature(order_id, payment_id, signature):
    if not RAZORPAY_KEY_SECRET:
        return False
    message = f"{order_id}|{payment_id}".encode("utf-8")
    expected = hmac.new(RAZORPAY_KEY_SECRET.encode("utf-8"), message, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, str(signature or ""))


def get_user_active_subscription(db, user_email):
    cursor = db.cursor()
    cursor.execute("SELECT id FROM users WHERE lower(email)=%s LIMIT 1", (str(user_email or "").strip().lower(),))
    user = cursor.fetchone()
    if not user:
        return None

    cursor.execute(
        """
        SELECT id, plan_code, plan_name, amount, currency, duration_days, status, starts_at, ends_at, created_at
        FROM user_subscriptions
        WHERE user_id=%s
        ORDER BY created_at DESC
        LIMIT 1
    """,
        (user["id"],),
    )
    row = cursor.fetchone()
    if not row:
        return None

    def fmt(dt_value):
        if hasattr(dt_value, "strftime"):
            return dt_value.strftime("%Y-%m-%d %H:%M:%S")
        return str(dt_value) if dt_value is not None else None

    return {
        "id": row["id"],
        "plan_code": row["plan_code"],
        "plan_name": row["plan_name"],
        "amount": float(row["amount"]),
        "currency": row["currency"],
        "duration_days": int(row["duration_days"]),
        "status": row["status"],
        "starts_at": fmt(row.get("starts_at")),
        "ends_at": fmt(row.get("ends_at")),
        "created_at": fmt(row.get("created_at")),
    }


# ---------------- DATABASE ----------------
def db_connect():
    if DATABASE_URL:
        return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DB_NAME,
        cursor_factory=RealDictCursor,
    )


def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = db_connect()
    return db


def get_db_connection():
    return db_connect()


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


def init_db():
    db = get_db_connection()
    cursor = db.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255),
            email VARCHAR(255) UNIQUE,
            password VARCHAR(255),
            phone VARCHAR(30),
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    cursor.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE")
    cursor.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS phone VARCHAR(30)")
    cursor.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS contacts (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255),
            email VARCHAR(255),
            message TEXT
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolios (
            id SERIAL PRIMARY KEY,
            user_id INT NOT NULL,
            name VARCHAR(100) NOT NULL DEFAULT 'Primary',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id SERIAL PRIMARY KEY,
            portfolio_id INT NOT NULL,
            snapshot_date DATE NOT NULL,
            invested_value DECIMAL(18, 2) NOT NULL DEFAULT 0,
            current_value DECIMAL(18, 2) NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (portfolio_id) REFERENCES portfolios(id) ON DELETE CASCADE,
            UNIQUE (portfolio_id, snapshot_date)
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS assets (
            id SERIAL PRIMARY KEY,
            portfolio_id INT NOT NULL,
            symbol VARCHAR(40) NOT NULL,
            asset_type VARCHAR(40) NOT NULL DEFAULT 'Equity',
            quantity DECIMAL(18, 4) NOT NULL DEFAULT 0,
            avg_price DECIMAL(18, 2) NOT NULL DEFAULT 0,
            current_price DECIMAL(18, 2) NOT NULL DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (portfolio_id) REFERENCES portfolios(id) ON DELETE CASCADE,
            UNIQUE (portfolio_id, symbol)
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id SERIAL PRIMARY KEY,
            portfolio_id INT NOT NULL,
            asset_id INT NOT NULL,
            txn_type VARCHAR(10) NOT NULL CHECK (txn_type IN ('BUY', 'SELL')),
            quantity DECIMAL(18, 4) NOT NULL,
            price DECIMAL(18, 2) NOT NULL,
            amount DECIMAL(18, 2) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (portfolio_id) REFERENCES portfolios(id) ON DELETE CASCADE,
            FOREIGN KEY (asset_id) REFERENCES assets(id) ON DELETE CASCADE
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS watchlist (
            id SERIAL PRIMARY KEY,
            portfolio_id INT NOT NULL,
            symbol VARCHAR(40) NOT NULL,
            asset_type VARCHAR(40) NOT NULL DEFAULT 'Equity',
            target_price DECIMAL(18, 2) NULL,
            note VARCHAR(255) NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (portfolio_id) REFERENCES portfolios(id) ON DELETE CASCADE,
            UNIQUE (portfolio_id, symbol)
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS stop_losses (
            id SERIAL PRIMARY KEY,
            portfolio_id INT NOT NULL,
            symbol VARCHAR(40) NOT NULL,
            stop_loss DECIMAL(18, 2) NOT NULL,
            is_triggered BOOLEAN NOT NULL DEFAULT FALSE,
            triggered_price DECIMAL(18, 2) NULL,
            triggered_at TIMESTAMP NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (portfolio_id) REFERENCES portfolios(id) ON DELETE CASCADE,
            UNIQUE (portfolio_id, symbol)
        )
    """
    )
    cursor.execute("ALTER TABLE stop_losses ADD COLUMN IF NOT EXISTS is_triggered BOOLEAN NOT NULL DEFAULT FALSE")
    cursor.execute("ALTER TABLE stop_losses ADD COLUMN IF NOT EXISTS triggered_price DECIMAL(18, 2) NULL")
    cursor.execute("ALTER TABLE stop_losses ADD COLUMN IF NOT EXISTS triggered_at TIMESTAMP NULL")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS recommendations (
            id SERIAL PRIMARY KEY,
            admin_email VARCHAR(255) NOT NULL,
            symbol VARCHAR(40) NOT NULL,
            action VARCHAR(10) NOT NULL CHECK (action IN ('BUY', 'HOLD', 'SELL')),
            entry_price DECIMAL(18, 2) NULL,
            target_price DECIMAL(18, 2) NULL,
            stop_loss DECIMAL(18, 2) NULL,
            note TEXT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS recommendation_targets (
            id SERIAL PRIMARY KEY,
            recommendation_id INT NOT NULL,
            user_id INT NOT NULL,
            delivered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (recommendation_id) REFERENCES recommendations(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            UNIQUE (recommendation_id, user_id)
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS market_reports (
            id SERIAL PRIMARY KEY,
            admin_email VARCHAR(255) NOT NULL,
            title VARCHAR(255) NOT NULL,
            summary TEXT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS market_report_targets (
            id SERIAL PRIMARY KEY,
            report_id INT NOT NULL,
            user_id INT NOT NULL,
            delivered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (report_id) REFERENCES market_reports(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            UNIQUE (report_id, user_id)
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS password_reset_otps (
            id SERIAL PRIMARY KEY,
            email VARCHAR(255) NOT NULL,
            otp_hash VARCHAR(255) NOT NULL,
            expires_at TIMESTAMP NOT NULL,
            used BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS user_subscriptions (
            id SERIAL PRIMARY KEY,
            user_id INT NOT NULL,
            plan_code VARCHAR(50) NOT NULL,
            plan_name VARCHAR(120) NOT NULL,
            amount DECIMAL(18, 2) NOT NULL,
            currency VARCHAR(10) NOT NULL DEFAULT 'INR',
            duration_days INT NOT NULL,
            status VARCHAR(20) NOT NULL DEFAULT 'CREATED',
            razorpay_order_id VARCHAR(120) NOT NULL UNIQUE,
            razorpay_payment_id VARCHAR(120) NULL UNIQUE,
            razorpay_signature VARCHAR(255) NULL,
            starts_at TIMESTAMP NULL,
            ends_at TIMESTAMP NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS blog_posts (
            id SERIAL PRIMARY KEY,
            author_email VARCHAR(255) NOT NULL,
            title VARCHAR(255) NOT NULL,
            excerpt TEXT NULL,
            content TEXT NOT NULL,
            is_published BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    db.commit()
    cursor.close()
    db.close()
    print("Database initialized!")


def get_or_create_portfolio_id(db, email):
    cursor = db.cursor()
    cursor.execute("SELECT id FROM users WHERE email=%s", (email,))
    user = cursor.fetchone()
    if not user:
        return None

    user_id = user["id"]
    cursor.execute("SELECT id FROM portfolios WHERE user_id=%s LIMIT 1", (user_id,))
    portfolio = cursor.fetchone()

    if portfolio:
        return portfolio["id"]

    cursor.execute("INSERT INTO portfolios (user_id, name) VALUES (%s, %s) RETURNING id", (user_id, "Primary"))
    portfolio_id = cursor.fetchone()["id"]
    db.commit()
    return portfolio_id


def get_user_profile(db, email):
    cursor = db.cursor()
    cursor.execute(
        """
        SELECT id, name, email, phone, created_at
        FROM users
        WHERE lower(email)=%s
        LIMIT 1
    """,
        (str(email or "").strip().lower(),),
    )
    row = cursor.fetchone()
    if not row:
        return None

    created = row.get("created_at")
    if hasattr(created, "strftime"):
        created_str = created.strftime("%d %b %Y")
    else:
        created_str = str(created) if created else "-"

    return {
        "id": row["id"],
        "name": row.get("name") or "",
        "email": row.get("email") or "",
        "phone": row.get("phone") or "",
        "created_at": created_str,
    }


def is_admin_email(email):
    e = str(email or "").strip().lower()
    return e in ADMIN_EMAILS


@app.before_request
def enforce_active_user_session():
    if "user" not in session:
        return None

    endpoint = request.endpoint or ""
    if endpoint == "static":
        return None

    email = str(session["user"].get("email", "")).strip().lower()
    if not email:
        session.clear()
        return redirect("/")

    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT is_active FROM users WHERE lower(email)=%s LIMIT 1", (email,))
        row = cursor.fetchone()
    except Exception:
        row = None

    if row and row.get("is_active", True):
        return None

    session.clear()
    json_paths = (
        "/admin/",
        "/watchlist/",
        "/stock/",
        "/order/",
        "/prices/",
        "/transactions/",
        "/rebalance/",
        "/market/",
        "/dashboard/",
    )
    if request.path.startswith(json_paths):
        return jsonify({"status": "error", "message": "Account is deactivated. Contact admin."}), 403
    return redirect("/")


def get_portfolio_snapshot(db, portfolio_id):
    cursor = db.cursor()
    cursor.execute(
        """
        SELECT symbol, asset_type, quantity, avg_price, current_price
        FROM assets
        WHERE portfolio_id=%s AND quantity > 0
        ORDER BY symbol ASC
    """,
        (portfolio_id,),
    )
    asset_rows = cursor.fetchall()

    holdings = []
    invested_total = 0.0
    current_total = 0.0
    type_values = {}

    for row in asset_rows:
        quantity = float(row["quantity"])
        avg_price = float(row["avg_price"])
        current_price = float(row["current_price"])
        invested = quantity * avg_price
        current = quantity * current_price
        asset_type = row["asset_type"]

        invested_total += invested
        current_total += current
        type_values[asset_type] = type_values.get(asset_type, 0.0) + current

        holdings.append(
            {
                "asset": row["symbol"],
                "type": asset_type,
                "quantity": round(quantity, 4),
                "avg_price": round(avg_price, 2),
                "current_price": round(current_price, 2),
                "invested": round(invested, 2),
                "current": round(current, 2),
                "pl": round(current - invested, 2),
            }
        )

    unrealized_pl = current_total - invested_total
    returns_pct = round((unrealized_pl / invested_total) * 100, 2) if invested_total > 0 else 0.0

    # Realized P/L is derived from executed BUY/SELL transactions using running average cost.
    cursor.execute(
        """
        SELECT a.symbol, t.txn_type, t.quantity, t.price
        FROM transactions t
        JOIN assets a ON a.id = t.asset_id
        WHERE t.portfolio_id=%s
        ORDER BY t.created_at ASC, t.id ASC
    """,
        (portfolio_id,),
    )
    txn_rows = cursor.fetchall()
    running = {}
    realized_pl = 0.0
    for tx in txn_rows:
        symbol = tx["symbol"]
        txn_type = str(tx["txn_type"]).upper()
        qty = float(tx["quantity"] or 0.0)
        price = float(tx["price"] or 0.0)
        if qty <= 0:
            continue

        state = running.get(symbol, {"qty": 0.0, "avg": 0.0})
        if txn_type == "BUY":
            new_qty = state["qty"] + qty
            if new_qty > 0:
                new_avg = ((state["qty"] * state["avg"]) + (qty * price)) / new_qty
            else:
                new_avg = 0.0
            running[symbol] = {"qty": new_qty, "avg": new_avg}
        else:  # SELL
            matched_qty = min(qty, state["qty"])
            realized_pl += matched_qty * (price - state["avg"])
            remaining_qty = max(0.0, state["qty"] - qty)
            running[symbol] = {"qty": remaining_qty, "avg": state["avg"] if remaining_qty > 0 else 0.0}

    cursor.execute(
        """
        SELECT MAX(updated_at) AS last_updated
        FROM assets
        WHERE portfolio_id=%s
    """,
        (portfolio_id,),
    )
    last_updated_row = cursor.fetchone()
    last_updated = last_updated_row.get("last_updated") if last_updated_row else None
    if hasattr(last_updated, "strftime"):
        last_updated = last_updated.strftime("%Y-%m-%d %H:%M:%S")
    elif last_updated is not None:
        last_updated = str(last_updated)
    else:
        last_updated = "-"

    risk = "Low"
    equity_value = type_values.get("Equity", 0.0)
    equity_ratio = (equity_value / current_total) if current_total > 0 else 0.0
    if equity_ratio >= 0.7:
        risk = "High"
    elif equity_ratio >= 0.4:
        risk = "Moderate"

    type_allocations = {}
    for asset_type in ["Equity", "Debt", "Gold", "Crypto"]:
        value = type_values.get(asset_type, 0.0)
        pct = (value / current_total * 100) if current_total > 0 else 0.0
        type_allocations[asset_type] = round(pct, 2)

    portfolio = {
        "invested": round(invested_total, 2),
        "current": round(current_total, 2),
        "returns": returns_pct,
        "unrealized_pl": round(unrealized_pl, 2),
        "realized_pl": round(realized_pl, 2),
        "total_pl": round(unrealized_pl + realized_pl, 2),
        "last_updated": last_updated,
        "risk": risk,
    }

    return portfolio, holdings, type_values, type_allocations


def fetch_watchlist_quote(symbol):
    if yf is None:
        return None, None

    try:
        ticker = yf.Ticker(symbol)
        history = ticker.history(period="5d")
        if history.empty:
            return None, None

        close_series = history["Close"].dropna()
        if close_series.empty:
            return None, None

        current_price = float(close_series.iloc[-1])
        prev_close = None
        if len(close_series) >= 2:
            prev_close = float(close_series.iloc[-2])
        else:
            info_prev = ticker.info.get("previousClose")
            if info_prev:
                prev_close = float(info_prev)

        return round(current_price, 2), round(prev_close, 2) if prev_close else None
    except Exception:
        return None, None


def get_symbol_market_snapshot(symbol):
    if yf is None:
        return None

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}

        intraday = ticker.history(period="1d", interval="1m")
        day_volume = 0.0
        avg_traded_price = None
        current_price = None

        if not intraday.empty:
            close_series = intraday["Close"].dropna()
            vol_series = intraday["Volume"].fillna(0)
            if not close_series.empty:
                current_price = float(close_series.iloc[-1])
            day_volume = float(vol_series.sum())
            vol_sum = float(vol_series.sum())
            if vol_sum > 0:
                avg_traded_price = float((intraday["Close"].fillna(0) * vol_series).sum() / vol_sum)

        if current_price is None:
            day = ticker.history(period="5d")
            if not day.empty:
                close_series = day["Close"].dropna()
                if not close_series.empty:
                    current_price = float(close_series.iloc[-1])

        prev_close = info.get("previousClose")
        if prev_close is None:
            day = ticker.history(period="5d")
            if not day.empty:
                close_series = day["Close"].dropna()
                if len(close_series) >= 2:
                    prev_close = float(close_series.iloc[-2])

        week52_low = info.get("fiftyTwoWeekLow")
        week52_high = info.get("fiftyTwoWeekHigh")
        if week52_low is None or week52_high is None:
            year = ticker.history(period="1y")
            if not year.empty:
                lows = year["Low"].dropna()
                highs = year["High"].dropna()
                if week52_low is None and not lows.empty:
                    week52_low = float(lows.min())
                if week52_high is None and not highs.empty:
                    week52_high = float(highs.max())

        if avg_traded_price is None and current_price is not None:
            avg_traded_price = current_price

        if current_price is None:
            return None

        bid_levels = []
        ask_levels = []
        base_qty = max(100, int(day_volume / 5000) if day_volume > 0 else 500)
        steps = [0.0015, 0.003, 0.0045, 0.006, 0.0075]
        for idx, step in enumerate(steps):
            bid_price = round(current_price * (1 - step), 2)
            ask_price = round(current_price * (1 + step), 2)
            bid_levels.append({"price": bid_price, "quantity": base_qty + (idx * 25)})
            ask_levels.append({"price": ask_price, "quantity": base_qty + (idx * 20)})

        performance_pct = None
        if prev_close and prev_close > 0:
            performance_pct = round(((current_price - float(prev_close)) / float(prev_close)) * 100, 2)

        return {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "previous_close": round(float(prev_close), 2) if prev_close else None,
            "performance_pct": performance_pct,
            "day_volume": int(day_volume),
            "avg_traded_price": round(float(avg_traded_price), 2) if avg_traded_price is not None else None,
            "week52_low": round(float(week52_low), 2) if week52_low is not None else None,
            "week52_high": round(float(week52_high), 2) if week52_high is not None else None,
            "order_book": {
                "bids": bid_levels,
                "asks": ask_levels,
            },
        }
    except Exception:
        return None


def get_stock_chart_data(symbol, timeframe):
    if yf is None:
        return None, "yfinance_not_installed"

    timeframe_map = {
        "15min": {"period": "5d", "interval": "15m"},
        "1hr": {"period": "1mo", "interval": "60m"},
        "1D": {"period": "6mo", "interval": "1d"},
        "1W": {"period": "2y", "interval": "1wk"},
        "1M": {"period": "5y", "interval": "1mo"},
    }

    config = timeframe_map.get(timeframe, timeframe_map["1D"])
    try:
        ticker = yf.Ticker(symbol)
        history = ticker.history(period=config["period"], interval=config["interval"])
        if history.empty:
            return None, "no_data"

        history = history.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        if history.empty:
            return None, "no_data"

        labels = []
        close_prices = []
        high_prices = []
        low_prices = []
        volumes = []
        for idx, row in history.iterrows():
            if hasattr(idx, "strftime"):
                if config["interval"] in {"15m", "60m"}:
                    labels.append(idx.strftime("%d %b %H:%M"))
                else:
                    labels.append(idx.strftime("%d %b %Y"))
            else:
                labels.append(str(idx))
            close_prices.append(round(float(row["Close"]), 2))
            high_prices.append(round(float(row["High"]), 2))
            low_prices.append(round(float(row["Low"]), 2))
            volumes.append(int(row["Volume"]))

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "labels": labels,
            "close_prices": close_prices,
            "high_prices": high_prices,
            "low_prices": low_prices,
            "volumes": volumes,
        }, None
    except Exception:
        return None, "fetch_failed"


def get_market_indices_snapshot():
    if yf is None:
        return None, "yfinance_not_installed"

    results = []
    for label, symbol in INDEX_SYMBOLS:
        try:
            ticker = yf.Ticker(symbol)
            price = None
            prev_close = None

            fast_info = getattr(ticker, "fast_info", None)
            if fast_info:
                last_price = fast_info.get("lastPrice")
                if last_price:
                    price = float(last_price)
                previous = fast_info.get("previousClose")
                if previous:
                    prev_close = float(previous)

            history = ticker.history(period="5d")
            if (price is None or price <= 0) and not history.empty:
                closes = history["Close"].dropna()
                if not closes.empty:
                    price = float(closes.iloc[-1])
                if prev_close is None and len(closes) >= 2:
                    prev_close = float(closes.iloc[-2])

            if price is None:
                continue

            change = None
            change_pct = None
            trend = "flat"
            if prev_close and prev_close > 0:
                change = round(price - prev_close, 2)
                change_pct = round((change / prev_close) * 100, 2)
                if change > 0:
                    trend = "up"
                elif change < 0:
                    trend = "down"

            results.append(
                {
                    "label": label,
                    "symbol": symbol,
                    "price": round(price, 2),
                    "change": change,
                    "change_pct": change_pct,
                    "trend": trend,
                }
            )
        except Exception:
            continue

    return results, None


def get_market_overview_snapshot():
    if yf is None:
        return None, "yfinance_not_installed"

    sectors = {}
    movers = []

    for sector, symbols in MARKET_HEATMAP_UNIVERSE.items():
        sector_rows = []
        for label, symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                price = None
                prev_close = None

                fast_info = getattr(ticker, "fast_info", None)
                if fast_info:
                    last_price = fast_info.get("lastPrice")
                    previous = fast_info.get("previousClose")
                    if last_price:
                        price = float(last_price)
                    if previous:
                        prev_close = float(previous)

                history = ticker.history(period="5d")
                if (price is None or price <= 0) and not history.empty:
                    closes = history["Close"].dropna()
                    if not closes.empty:
                        price = float(closes.iloc[-1])
                    if prev_close is None and len(closes) >= 2:
                        prev_close = float(closes.iloc[-2])

                if price is None or price <= 0:
                    continue

                change_pct = None
                trend = "flat"
                if prev_close and prev_close > 0:
                    change_pct = round(((price - prev_close) / prev_close) * 100, 2)
                    if change_pct > 0:
                        trend = "up"
                    elif change_pct < 0:
                        trend = "down"

                sparkline = []
                spark_hist = ticker.history(period="1mo", interval="1d")
                if not spark_hist.empty:
                    close_series = spark_hist["Close"].dropna().tail(20)
                    sparkline = [round(float(v), 2) for v in close_series.tolist()]

                row = {
                    "sector": sector,
                    "label": label,
                    "symbol": symbol,
                    "price": round(price, 2),
                    "change_pct": change_pct,
                    "trend": trend,
                    "sparkline": sparkline,
                }
                sector_rows.append(row)
                if change_pct is not None:
                    movers.append(row)
            except Exception:
                continue

        if sector_rows:
            sectors[sector] = sorted(
                sector_rows,
                key=lambda item: item["change_pct"] if item["change_pct"] is not None else -999.0,
                reverse=True,
            )

    movers.sort(key=lambda item: abs(item["change_pct"]), reverse=True)
    return {"sectors": sectors, "top_movers": movers[:12]}, None


def _parse_epoch_to_utc_string(epoch_value):
    try:
        ts = int(epoch_value)
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return None


def _extract_next_earnings_date(ticker):
    try:
        calendar = getattr(ticker, "calendar", None)
        if calendar is not None:
            if hasattr(calendar, "to_dict"):
                cal_dict = calendar.to_dict()
                value = cal_dict.get("Earnings Date")
                if isinstance(value, list) and value:
                    value = value[0]
                if hasattr(value, "to_pydatetime"):
                    value = value.to_pydatetime()
                if hasattr(value, "strftime"):
                    return value.strftime("%Y-%m-%d")
            elif isinstance(calendar, dict):
                value = calendar.get("Earnings Date")
                if isinstance(value, list) and value:
                    value = value[0]
                if hasattr(value, "strftime"):
                    return value.strftime("%Y-%m-%d")
    except Exception:
        pass

    try:
        earnings_df = ticker.get_earnings_dates(limit=1)
        if earnings_df is not None and not earnings_df.empty:
            idx_value = earnings_df.index[0]
            if hasattr(idx_value, "to_pydatetime"):
                idx_value = idx_value.to_pydatetime()
            if hasattr(idx_value, "strftime"):
                return idx_value.strftime("%Y-%m-%d")
    except Exception:
        return None
    return None


def get_trending_market_news(limit=12):
    queries = [
        "india stock market",
        "nifty 50",
        "sensex",
        "nse bse stocks",
        "rbi policy repo rate",
        "india cpi inflation",
        "india gdp growth",
        "fii dii flows india",
        "bank nifty",
        "nifty it pharma auto",
    ]
    india_focus_symbols = [
        "^NSEI",       # NIFTY 50
        "^BSESN",      # SENSEX
        "^NSEBANK",    # BANK NIFTY
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "BHARTIARTL.NS",
        "LT.NS",
        "ITC.NS",
    ]
    seen_urls = set()
    items = []

    for query in queries:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={quote_plus(query)}&quotesCount=0&newsCount=10"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urlopen(req, timeout=6) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            continue

        for entry in payload.get("news", []):
            link = (
                entry.get("link")
                or entry.get("canonicalUrl", {}).get("url")
                or entry.get("url")
                or ""
            )
            if not link or link in seen_urls:
                continue
            seen_urls.add(link)

            title = entry.get("title") or "Untitled"
            source = entry.get("publisher") or entry.get("provider") or "Market News"
            pub_raw = entry.get("providerPublishTime") or entry.get("pubDate")
            published_at = _parse_epoch_to_utc_string(pub_raw) if pub_raw else None
            if isinstance(pub_raw, str) and not published_at:
                published_at = pub_raw

            items.append(
                {
                    "symbol": "MARKET",
                    "title": str(title),
                    "source": str(source),
                    "url": str(link),
                    "published_at": published_at or "-",
                }
            )
            if len(items) >= limit:
                return items

    # Add symbol-led Indian headlines (companies + index/economic context)
    if yf is not None and len(items) < limit:
        for symbol in india_focus_symbols:
            try:
                ticker = yf.Ticker(symbol)
                news_list = getattr(ticker, "news", []) or []
                for entry in news_list[:6]:
                    link = (
                        entry.get("link")
                        or entry.get("canonicalUrl", {}).get("url")
                        or entry.get("url")
                        or ""
                    )
                    if not link or link in seen_urls:
                        continue
                    seen_urls.add(link)

                    title = entry.get("title") or "Untitled"
                    source = entry.get("publisher") or entry.get("provider") or "Market News"
                    pub_raw = entry.get("providerPublishTime") or entry.get("pubDate")
                    published_at = _parse_epoch_to_utc_string(pub_raw) if pub_raw else None
                    if isinstance(pub_raw, str) and not published_at:
                        published_at = pub_raw

                    items.append(
                        {
                            "symbol": symbol.replace(".NS", ""),
                            "title": str(title),
                            "source": str(source),
                            "url": str(link),
                            "published_at": published_at or "-",
                        }
                    )
                    if len(items) >= limit:
                        return items
            except Exception:
                continue

    return items


def get_watchlist_news_and_calendar(db, portfolio_id):
    cursor = db.cursor()
    cursor.execute("SELECT symbol FROM watchlist WHERE portfolio_id=%s ORDER BY created_at DESC LIMIT 8", (portfolio_id,))
    rows = cursor.fetchall()
    symbols = [str(r["symbol"]).strip().upper() for r in rows if r.get("symbol")]

    news_items = []
    calendar_events = []

    fed_events = DEFAULT_FED_EVENTS
    try:
        configured_fed = os.getenv("FED_EVENTS_JSON", "").strip()
        if configured_fed:
            parsed = json.loads(configured_fed)
            if isinstance(parsed, list) and parsed:
                fed_events = parsed
    except Exception:
        pass

    ipo_events = []
    try:
        configured_ipo = os.getenv("IPO_EVENTS_JSON", "").strip()
        if configured_ipo:
            parsed = json.loads(configured_ipo)
            if isinstance(parsed, list):
                ipo_events = parsed
    except Exception:
        pass

    for item in fed_events:
        event_date = str(item.get("date", "")).strip()
        title = str(item.get("title", "Fed Event")).strip()
        if event_date:
            calendar_events.append(
                {"date": event_date, "title": title, "type": str(item.get("type", "Fed")).strip() or "Fed", "symbol": ""}
            )

    for item in ipo_events:
        event_date = str(item.get("date", "")).strip()
        title = str(item.get("title", "IPO Event")).strip()
        symbol = str(item.get("symbol", "")).strip().upper()
        if event_date:
            calendar_events.append(
                {"date": event_date, "title": title, "type": "IPO", "symbol": symbol}
            )

    trending_news = get_trending_market_news(limit=12)

    if yf is None:
        calendar_events.sort(key=lambda x: x.get("date", "9999-12-31"))
        return [], trending_news, calendar_events[:20], symbols, "yfinance_not_installed"

    for symbol in symbols[:6]:
        try:
            ticker = yf.Ticker(symbol)

            earnings_date = _extract_next_earnings_date(ticker)
            if earnings_date:
                calendar_events.append(
                    {
                        "date": earnings_date,
                        "title": f"{symbol} earnings window",
                        "type": "Earnings",
                        "symbol": symbol,
                    }
                )

            news_list = getattr(ticker, "news", []) or []
            for entry in news_list[:4]:
                link = (
                    entry.get("link")
                    or entry.get("canonicalUrl", {}).get("url")
                    or entry.get("url")
                    or ""
                )
                title = entry.get("title") or "Untitled"
                source = entry.get("publisher") or entry.get("provider") or "News"
                pub_raw = entry.get("providerPublishTime") or entry.get("pubDate")
                published_at = _parse_epoch_to_utc_string(pub_raw) if pub_raw else None
                if isinstance(pub_raw, str) and not published_at:
                    published_at = pub_raw

                if not link:
                    continue

                news_items.append(
                    {
                        "symbol": symbol,
                        "title": str(title),
                        "source": str(source),
                        "url": str(link),
                        "published_at": published_at or "-",
                    }
                )
        except Exception:
            continue

    seen_urls = set()
    dedup_news = []
    for item in news_items:
        url = item["url"]
        if url in seen_urls:
            continue
        seen_urls.add(url)
        dedup_news.append(item)
        if len(dedup_news) >= 16:
            break

    today = datetime.utcnow().date()
    normalized_events = []
    for event in calendar_events:
        raw_date = str(event.get("date", "")).strip()
        dt_obj = None
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y"):
            try:
                dt_obj = datetime.strptime(raw_date, fmt).date()
                break
            except Exception:
                continue
        if dt_obj:
            if dt_obj >= today:
                normalized_events.append({**event, "date": dt_obj.strftime("%Y-%m-%d")})
        elif raw_date:
            normalized_events.append(event)

    normalized_events.sort(key=lambda x: x.get("date", "9999-12-31"))
    return dedup_news, trending_news, normalized_events[:20], symbols, None


def search_stock_suggestions(query):
    query = str(query or "").strip()
    if len(query) < 2:
        return []

    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={quote_plus(query)}&quotesCount=10&newsCount=0"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urlopen(req, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []

    items = []
    for quote in payload.get("quotes", []):
        symbol = quote.get("symbol")
        name = quote.get("shortname") or quote.get("longname") or symbol
        quote_type = (quote.get("quoteType") or "").upper()
        exch = quote.get("exchange") or quote.get("exchDisp") or ""

        if not symbol:
            continue
        if quote_type not in {"EQUITY", "ETF", "INDEX", "MUTUALFUND"}:
            continue

        items.append(
            {
                "symbol": symbol.upper(),
                "name": name,
                "type": quote_type,
                "exchange": exch,
            }
        )

    # Deduplicate by symbol and keep first entries
    seen = set()
    unique = []
    for item in items:
        key = item["symbol"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= 8:
            break
    return unique


def resolve_market_symbol(symbol_input):
    symbol_input = str(symbol_input or "").strip()
    if not symbol_input:
        return ""

    raw_upper = symbol_input.upper()

    # If it already looks like a market symbol, keep it.
    if " " not in raw_upper and (".NS" in raw_upper or ".BO" in raw_upper or raw_upper.isalnum() or "^" in raw_upper):
        return raw_upper

    # Otherwise try resolving by Yahoo search and prefer Indian cash market symbols.
    suggestions = search_stock_suggestions(symbol_input)
    if not suggestions:
        return raw_upper

    preferred = []
    fallback = []
    for item in suggestions:
        sym = str(item.get("symbol", "")).strip().upper()
        if not sym:
            continue
        if sym.endswith(".NS") or sym.endswith(".BO"):
            preferred.append(sym)
        else:
            fallback.append(sym)

    if preferred:
        return preferred[0]
    if fallback:
        return fallback[0]
    return raw_upper


def get_watchlist(db, portfolio_id):
    cursor = db.cursor()
    cursor.execute(
        """
        SELECT w.id, w.symbol, w.asset_type, w.target_price, w.note, w.created_at,
               a.current_price
        FROM watchlist w
        LEFT JOIN assets a
          ON a.portfolio_id = w.portfolio_id
         AND a.symbol = w.symbol
        WHERE w.portfolio_id=%s
        ORDER BY w.created_at DESC
    """,
        (portfolio_id,),
    )
    rows = cursor.fetchall()
    items = []
    for row in rows:
        created = row.get("created_at")
        if hasattr(created, "strftime"):
            created_str = created.strftime("%Y-%m-%d %H:%M:%S")
        else:
            created_str = str(created)

        live_price, prev_close = fetch_watchlist_quote(row["symbol"])
        current_price = live_price if live_price is not None else row["current_price"]
        performance_pct = None
        trend = "flat"
        if current_price is not None and prev_close and prev_close > 0:
            performance_pct = round(((float(current_price) - float(prev_close)) / float(prev_close)) * 100, 2)
            if performance_pct > 0:
                trend = "up"
            elif performance_pct < 0:
                trend = "down"

        items.append(
            {
                "id": row["id"],
                "symbol": row["symbol"],
                "asset_type": row["asset_type"],
                "target_price": float(row["target_price"]) if row["target_price"] is not None else None,
                "current_price": float(current_price) if current_price is not None else None,
                "prev_close": float(prev_close) if prev_close is not None else None,
                "performance_pct": performance_pct,
                "trend": trend,
                "note": row["note"] or "",
                "created_at": created_str,
            }
        )
    return items


def get_user_recommendations(db, email, limit=20):
    cursor = db.cursor()
    cursor.execute("SELECT id FROM users WHERE email=%s LIMIT 1", (email,))
    user = cursor.fetchone()
    if not user:
        return []

    safe_limit = max(1, min(int(limit), 100))
    cursor.execute(
        """
        SELECT
            r.id,
            r.symbol,
            r.action,
            r.entry_price,
            r.target_price,
            r.stop_loss,
            r.note,
            r.created_at,
            r.admin_email
        FROM recommendation_targets rt
        JOIN recommendations r ON r.id = rt.recommendation_id
        WHERE rt.user_id=%s
        ORDER BY r.created_at DESC
        LIMIT %s
    """,
        (user["id"], safe_limit),
    )
    rows = cursor.fetchall()
    items = []
    for row in rows:
        created = row.get("created_at")
        if hasattr(created, "strftime"):
            created = created.strftime("%Y-%m-%d %H:%M:%S")
        items.append(
            {
                "id": row["id"],
                "symbol": row["symbol"],
                "action": row["action"],
                "entry_price": float(row["entry_price"]) if row["entry_price"] is not None else None,
                "target_price": float(row["target_price"]) if row["target_price"] is not None else None,
                "stop_loss": float(row["stop_loss"]) if row["stop_loss"] is not None else None,
                "note": row["note"] or "",
                "created_at": str(created),
                "admin_email": row["admin_email"],
            }
        )
    return items


def get_admin_recommendations(db, limit=30):
    safe_limit = max(1, min(int(limit), 100))
    cursor = db.cursor()
    cursor.execute(
        """
        SELECT
            r.id,
            r.symbol,
            r.action,
            r.entry_price,
            r.target_price,
            r.stop_loss,
            r.note,
            r.created_at,
            COUNT(rt.user_id) AS target_count
        FROM recommendations r
        LEFT JOIN recommendation_targets rt ON rt.recommendation_id = r.id
        GROUP BY r.id
        ORDER BY r.created_at DESC
        LIMIT %s
    """,
        (safe_limit,),
    )
    rows = cursor.fetchall()
    items = []
    for row in rows:
        created = row.get("created_at")
        if hasattr(created, "strftime"):
            created = created.strftime("%Y-%m-%d %H:%M:%S")
        items.append(
            {
                "id": row["id"],
                "symbol": row["symbol"],
                "action": row["action"],
                "entry_price": float(row["entry_price"]) if row["entry_price"] is not None else None,
                "target_price": float(row["target_price"]) if row["target_price"] is not None else None,
                "stop_loss": float(row["stop_loss"]) if row["stop_loss"] is not None else None,
                "note": row["note"] or "",
                "created_at": str(created),
                "target_count": int(row["target_count"] or 0),
            }
        )
    return items


def get_user_market_reports(db, email, limit=20):
    cursor = db.cursor()
    cursor.execute("SELECT id FROM users WHERE email=%s LIMIT 1", (email,))
    user = cursor.fetchone()
    if not user:
        return []

    safe_limit = max(1, min(int(limit), 100))
    cursor.execute(
        """
        SELECT
            mr.id,
            mr.title,
            mr.summary,
            mr.content,
            mr.created_at,
            mr.admin_email
        FROM market_report_targets mrt
        JOIN market_reports mr ON mr.id = mrt.report_id
        WHERE mrt.user_id=%s
        ORDER BY mr.created_at DESC
        LIMIT %s
    """,
        (user["id"], safe_limit),
    )
    rows = cursor.fetchall()
    items = []
    for row in rows:
        created = row.get("created_at")
        if hasattr(created, "strftime"):
            created = created.strftime("%Y-%m-%d %H:%M:%S")
        items.append(
            {
                "id": row["id"],
                "title": row["title"],
                "summary": row["summary"] or "",
                "content": row["content"] or "",
                "created_at": str(created),
                "admin_email": row["admin_email"],
            }
        )
    return items


def get_admin_market_reports(db, limit=30):
    safe_limit = max(1, min(int(limit), 100))
    cursor = db.cursor()
    cursor.execute(
        """
        SELECT
            mr.id,
            mr.title,
            mr.summary,
            mr.content,
            mr.created_at,
            COUNT(mrt.user_id) AS target_count
        FROM market_reports mr
        LEFT JOIN market_report_targets mrt ON mrt.report_id = mr.id
        GROUP BY mr.id
        ORDER BY mr.created_at DESC
        LIMIT %s
    """,
        (safe_limit,),
    )
    rows = cursor.fetchall()
    items = []
    for row in rows:
        created = row.get("created_at")
        if hasattr(created, "strftime"):
            created = created.strftime("%Y-%m-%d %H:%M:%S")
        items.append(
            {
                "id": row["id"],
                "title": row["title"],
                "summary": row["summary"] or "",
                "content": row["content"] or "",
                "created_at": str(created),
                "target_count": int(row["target_count"] or 0),
            }
        )
    return items


def get_blog_posts(db, published_only=True, limit=20):
    safe_limit = max(1, min(int(limit), 100))
    cursor = db.cursor()
    if published_only:
        cursor.execute(
            """
            SELECT id, author_email, title, excerpt, content, is_published, created_at
            FROM blog_posts
            WHERE is_published=TRUE
            ORDER BY created_at DESC
            LIMIT %s
        """,
            (safe_limit,),
        )
    else:
        cursor.execute(
            """
            SELECT id, author_email, title, excerpt, content, is_published, created_at
            FROM blog_posts
            ORDER BY created_at DESC
            LIMIT %s
        """,
            (safe_limit,),
        )

    rows = cursor.fetchall()
    items = []
    for row in rows:
        created = row.get("created_at")
        if hasattr(created, "strftime"):
            created_str = created.strftime("%d %b %Y")
        else:
            created_str = str(created)

        items.append(
            {
                "id": row["id"],
                "author_email": row["author_email"],
                "title": row["title"],
                "excerpt": row["excerpt"] or "",
                "content": row["content"] or "",
                "is_published": bool(row["is_published"]),
                "created_at": created_str,
            }
        )
    return items


def get_blog_post_by_id(db, blog_id):
    cursor = db.cursor()
    cursor.execute(
        """
        SELECT id, author_email, title, excerpt, content, is_published, created_at
        FROM blog_posts
        WHERE id=%s
        LIMIT 1
    """,
        (blog_id,),
    )
    row = cursor.fetchone()
    if not row:
        return None

    created = row.get("created_at")
    if hasattr(created, "strftime"):
        created_str = created.strftime("%d %b %Y")
    else:
        created_str = str(created)

    return {
        "id": row["id"],
        "author_email": row["author_email"],
        "title": row["title"],
        "excerpt": row["excerpt"] or "",
        "content": row["content"] or "",
        "is_published": bool(row["is_published"]),
        "created_at": created_str,
    }


def get_portfolio_performance_series(db, portfolio_id, days=60):
    cursor = db.cursor()
    safe_days = max(1, int(days))
    cutoff = datetime.utcnow().date() - timedelta(days=safe_days - 1)

    cursor.execute(
        """
        SELECT snapshot_date, invested_value, current_value
        FROM portfolio_snapshots
        WHERE portfolio_id=%s AND snapshot_date >= %s
        ORDER BY snapshot_date ASC
    """,
        (portfolio_id, cutoff),
    )
    rows = cursor.fetchall()
    if rows:
        labels = []
        net_invested = []
        current_value = []
        daily_flow = []
        prev_invested = None

        for row in rows:
            day = row.get("snapshot_date")
            if hasattr(day, "strftime"):
                labels.append(day.strftime("%Y-%m-%d"))
            else:
                labels.append(str(day))

            invested = float(row.get("invested_value") or 0)
            current = float(row.get("current_value") or 0)
            net_invested.append(round(invested, 2))
            current_value.append(round(current, 2))

            if prev_invested is None:
                daily_flow.append(round(invested, 2))
            else:
                daily_flow.append(round(invested - prev_invested, 2))
            prev_invested = invested

        return {
            "labels": labels,
            "net_invested": net_invested,
            "current_value": current_value,
            "daily_flow": daily_flow,
        }

    # Fallback for legacy data before snapshots were introduced.
    cursor = db.cursor()
    cursor.execute(
        """
        SELECT txn_type, amount, created_at
        FROM transactions
        WHERE portfolio_id=%s
        ORDER BY created_at ASC
    """,
        (portfolio_id,),
    )
    rows = cursor.fetchall()
    if not rows:
        return {"labels": [], "net_invested": [], "current_value": [], "daily_flow": []}

    daily = defaultdict(float)
    for row in rows:
        created = row.get("created_at")
        if not hasattr(created, "date"):
            continue
        day = created.date()
        if day < cutoff:
            continue
        amount = float(row.get("amount") or 0)
        if str(row.get("txn_type", "")).upper() == "BUY":
            daily[day] += amount
        else:
            daily[day] -= amount

    if not daily:
        return {"labels": [], "net_invested": [], "current_value": [], "daily_flow": []}

    sorted_days = sorted(daily.keys())
    labels = [d.strftime("%Y-%m-%d") for d in sorted_days]
    daily_flow = [round(daily[d], 2) for d in sorted_days]

    running = 0.0
    net_invested = []
    for value in daily_flow:
        running += value
        net_invested.append(round(running, 2))

    return {"labels": labels, "net_invested": net_invested, "current_value": [], "daily_flow": daily_flow}


def save_daily_portfolio_snapshot(db, portfolio_id, invested_value, current_value):
    cursor = db.cursor()
    today = datetime.utcnow().date()
    cursor.execute(
        """
        INSERT INTO portfolio_snapshots (portfolio_id, snapshot_date, invested_value, current_value)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (portfolio_id, snapshot_date)
        DO UPDATE SET
            invested_value = EXCLUDED.invested_value,
            current_value = EXCLUDED.current_value
    """,
        (portfolio_id, today, float(invested_value or 0), float(current_value or 0)),
    )
    db.commit()


def get_user_recommendation_performance(db, email, limit=12):
    cursor = db.cursor()
    cursor.execute("SELECT id FROM users WHERE email=%s LIMIT 1", (email,))
    user = cursor.fetchone()
    if not user:
        return {"items": [], "avg_return": 0.0, "hit_rate": 0.0}

    safe_limit = max(1, min(int(limit), 30))
    cursor.execute(
        """
        SELECT r.symbol, r.action, r.entry_price, r.target_price, r.stop_loss, r.created_at
        FROM recommendation_targets rt
        JOIN recommendations r ON r.id = rt.recommendation_id
        WHERE rt.user_id=%s
          AND r.entry_price IS NOT NULL
        ORDER BY r.created_at DESC
        LIMIT %s
    """,
        (user["id"], safe_limit),
    )
    rows = cursor.fetchall()

    items = []
    for row in rows:
        symbol = str(row["symbol"] or "").strip().upper()
        resolved_symbol = resolve_market_symbol(symbol)
        action = str(row["action"]).upper()
        entry_price = float(row["entry_price"]) if row["entry_price"] is not None else None
        if not resolved_symbol or not entry_price or entry_price <= 0:
            continue

        current_price, _ = fetch_watchlist_quote(resolved_symbol)
        if current_price is None:
            continue

        if action in {"BUY", "HOLD"}:
            return_pct = ((float(current_price) - entry_price) / entry_price) * 100
        else:  # SELL call
            return_pct = ((entry_price - float(current_price)) / entry_price) * 100

        items.append(
            {
                "symbol": resolved_symbol,
                "action": action,
                "entry_price": round(entry_price, 2),
                "current_price": round(float(current_price), 2),
                "return_pct": round(return_pct, 2),
            }
        )

    if not items:
        return {"items": [], "avg_return": 0.0, "hit_rate": 0.0}

    avg_return = round(sum(i["return_pct"] for i in items) / len(items), 2)
    hit_rate = round((sum(1 for i in items if i["return_pct"] > 0) / len(items)) * 100, 2)
    return {"items": items, "avg_return": avg_return, "hit_rate": hit_rate}


def process_stop_loss_triggers(db, portfolio_id, user_email):
    cursor = db.cursor()
    cursor.execute(
        """
        SELECT
            sl.id,
            sl.symbol,
            sl.stop_loss,
            sl.is_triggered,
            a.current_price
        FROM stop_losses sl
        LEFT JOIN assets a
          ON a.portfolio_id = sl.portfolio_id
         AND a.symbol = sl.symbol
        WHERE sl.portfolio_id=%s
    """,
        (portfolio_id,),
    )
    rows = cursor.fetchall()
    triggered_alerts = []

    for row in rows:
        symbol = row["symbol"]
        stop_loss = float(row["stop_loss"] or 0.0)
        current_price = row.get("current_price")
        is_triggered = bool(row.get("is_triggered"))

        if current_price is None or stop_loss <= 0:
            continue
        current_price = float(current_price)

        if not is_triggered and current_price <= stop_loss:
            trigger_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            email_ok, email_info = send_stop_loss_alert_email(
                user_email,
                symbol,
                round(stop_loss, 2),
                round(current_price, 2),
                trigger_time,
            )
            cursor.execute(
                """
                UPDATE stop_losses
                SET is_triggered=TRUE,
                    triggered_price=%s,
                    triggered_at=NOW(),
                    updated_at=NOW()
                WHERE id=%s
            """,
                (round(current_price, 2), row["id"]),
            )
            triggered_alerts.append(
                {
                    "symbol": symbol,
                    "stop_loss": round(stop_loss, 2),
                    "triggered_price": round(current_price, 2),
                    "email_sent": bool(email_ok),
                    "email_info": "sent" if email_ok else str(email_info),
                }
            )
            if not email_ok:
                print(f"[STOP LOSS EMAIL ERROR] {symbol} {user_email}: {email_info}")

    return triggered_alerts


def format_transaction_row(row):
    created = row.get("created_at")
    if hasattr(created, "strftime"):
        created_str = created.strftime("%Y-%m-%d %H:%M:%S")
    else:
        created_str = str(created)

    return {
        "id": row["id"],
        "symbol": row["symbol"],
        "asset_type": row["asset_type"],
        "txn_type": row["txn_type"],
        "quantity": float(row["quantity"]),
        "price": float(row["price"]),
        "amount": float(row["amount"]),
        "created_at": created_str,
    }


def get_transaction_history(db, portfolio_id, txn_type=None, symbol=None, date_from=None, date_to=None, limit=20):
    limit = max(1, min(int(limit), 100))

    base_query = """
        SELECT t.id, a.symbol, a.asset_type, t.txn_type, t.quantity, t.price, t.amount, t.created_at
        FROM transactions t
        JOIN assets a ON a.id = t.asset_id
        WHERE t.portfolio_id=%s
    """

    params = [portfolio_id]

    if txn_type:
        base_query += " AND t.txn_type=%s"
        params.append(txn_type.upper())

    if symbol:
        base_query += " AND a.symbol=%s"
        params.append(symbol.upper())

    if date_from:
        base_query += " AND DATE(t.created_at) >= %s"
        params.append(date_from)

    if date_to:
        base_query += " AND DATE(t.created_at) <= %s"
        params.append(date_to)

    base_query += f" ORDER BY t.created_at DESC LIMIT {limit}"

    cursor = db.cursor()
    cursor.execute(base_query, tuple(params))
    rows = cursor.fetchall()
    return [format_transaction_row(row) for row in rows]


def fetch_live_prices(symbols):
    if yf is None:
        return {}, symbols, "yfinance_not_installed"

    prices = {}
    failed = []

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            price = None

            fast_info = getattr(ticker, "fast_info", None)
            if fast_info:
                last_price = fast_info.get("lastPrice")
                if last_price:
                    price = float(last_price)

            if not price:
                history = ticker.history(period="5d")
                if not history.empty:
                    close_series = history["Close"].dropna()
                    if not close_series.empty:
                        price = float(close_series.iloc[-1])

            if price and price > 0:
                prices[symbol] = round(price, 2)
            else:
                failed.append(symbol)
        except Exception:
            failed.append(symbol)

    return prices, failed, None


# ---------------- ROUTES ----------------
@app.route("/")
def home():
    user = None
    db = get_db()
    blogs = get_blog_posts(db, published_only=True, limit=12)
    if "user" in session:
        user = session["user"]
    return render_template("index.html", user=user, blogs=blogs)


@app.route("/blog/<int:blog_id>")
def blog_detail(blog_id):
    db = get_db()
    blog = get_blog_post_by_id(db, blog_id)
    if not blog:
        return redirect("/")

    # Only published blogs are public. Admin can preview drafts.
    if not blog["is_published"]:
        if "user" not in session or not session["user"].get("is_admin"):
            return redirect("/")

    user = session["user"] if "user" in session else None
    return render_template("blog_detail.html", blog=blog, user=user)


@app.route("/login", methods=["POST"])
def login():
    data = request.json or {}
    email = data.get("email")
    password = data.get("password")

    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
    user = cursor.fetchone()

    if user and not user.get("is_active", True):
        return jsonify({"status": "error", "message": "Your account is deactivated. Contact admin."}), 403

    if user and check_password_hash(user["password"], password):
        admin_flag = is_admin_email(user["email"])
        session["user"] = {
            "name": user["name"],
            "email": user["email"],
            "is_admin": admin_flag,
        }
        return jsonify(
            {
                "status": "success",
                "message": f"Welcome {user['name']}!",
                "redirect_url": "/admin" if admin_flag else "/dashboard",
            }
        )

    return jsonify({"status": "error", "message": "Invalid email or password!"})


@app.route("/admin/login", methods=["POST"])
def admin_login():
    data = request.json or {}
    email = data.get("email")
    password = data.get("password")

    if not is_admin_email(email):
        return jsonify({"status": "error", "message": "This account is not authorized for admin login"}), 403

    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
    user = cursor.fetchone()

    if not user:
        return jsonify({"status": "error", "message": "Invalid admin credentials"}), 401
    if not user.get("is_active", True):
        return jsonify({"status": "error", "message": "Admin account is deactivated"}), 403
    if not check_password_hash(user["password"], password):
        return jsonify({"status": "error", "message": "Invalid admin credentials"}), 401

    session["user"] = {
        "name": user["name"],
        "email": user["email"],
        "is_admin": True,
    }
    return jsonify({"status": "success", "message": f"Welcome Admin {user['name']}!", "redirect_url": "/admin"})


@app.route('/userlog', methods=['POST'])
def userlog():
    name = request.form['name']
    password = request.form['password']

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT password FROM users WHERE name=%s", (name,))
    result = cur.fetchone()

    cur.close()
    conn.close()

    if result and check_password_hash(result["password"], password):
        return render_template('userlog.html')

    return render_template('index.html', msg='Incorrect Credentials')


@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")

    db = get_db()
    portfolio_id = get_or_create_portfolio_id(db, session["user"]["email"])
    if not portfolio_id:
        return redirect("/")

    portfolio, holdings, _, type_allocations = get_portfolio_snapshot(db, portfolio_id)
    save_daily_portfolio_snapshot(db, portfolio_id, portfolio.get("invested", 0), portfolio.get("current", 0))
    transactions = get_transaction_history(db, portfolio_id, limit=15)
    watchlist = get_watchlist(db, portfolio_id)
    watch_news, trending_news, economic_events, watch_symbols, insights_error = get_watchlist_news_and_calendar(db, portfolio_id)
    recommendations = get_user_recommendations(db, session["user"]["email"], limit=20)
    market_reports = get_user_market_reports(db, session["user"]["email"], limit=20)
    user_profile = get_user_profile(db, session["user"]["email"])
    portfolio_perf = get_portfolio_performance_series(db, portfolio_id, days=60)
    rec_perf = get_user_recommendation_performance(db, session["user"]["email"], limit=12)
    active_subscription = get_user_active_subscription(db, session["user"]["email"])

    return render_template(
        "dashboard.html",
        user=session["user"],
        portfolio=portfolio,
        holdings=holdings,
        transactions=transactions,
        type_allocations=type_allocations,
        watchlist=watchlist,
        watch_news=watch_news,
        trending_news=trending_news,
        economic_events=economic_events,
        insight_symbols=watch_symbols,
        insights_error=insights_error,
        user_profile=user_profile,
        recommendations=recommendations,
        market_reports=market_reports,
        portfolio_perf=portfolio_perf,
        recommendation_perf=rec_perf,
        active_subscription=active_subscription,
    )


@app.route("/dashboard/profile")
def profile_dashboard():
    if "user" not in session:
        return redirect("/")

    db = get_db()
    user_profile = get_user_profile(db, session["user"]["email"])
    if not user_profile:
        return redirect("/dashboard")
    active_subscription = get_user_active_subscription(db, session["user"]["email"])

    return render_template(
        "profile_dashboard.html",
        user=session["user"],
        user_profile=user_profile,
        active_subscription=active_subscription,
    )


@app.route("/dashboard/performance")
def performance_dashboard():
    if "user" not in session:
        return redirect("/")

    db = get_db()
    portfolio_id = get_or_create_portfolio_id(db, session["user"]["email"])
    if not portfolio_id:
        return redirect("/dashboard")

    portfolio, _, _, _ = get_portfolio_snapshot(db, portfolio_id)
    save_daily_portfolio_snapshot(db, portfolio_id, portfolio.get("invested", 0), portfolio.get("current", 0))
    portfolio_perf = get_portfolio_performance_series(db, portfolio_id, days=180)
    rec_perf = get_user_recommendation_performance(db, session["user"]["email"], limit=20)

    return render_template(
        "performance_dashboard.html",
        user=session["user"],
        portfolio=portfolio,
        portfolio_perf=portfolio_perf,
        recommendation_perf=rec_perf,
    )


@app.route("/admin")
def admin_dashboard():
    if "user" not in session:
        return redirect("/")

    if not session["user"].get("is_admin"):
        return redirect("/dashboard")

    db = get_db()
    cursor = db.cursor()

    cursor.execute("SELECT COUNT(*) AS total_users FROM users")
    total_users = cursor.fetchone()["total_users"]

    cursor.execute(
        """
        SELECT COUNT(*) AS total_transactions
        FROM transactions
    """
    )
    total_transactions = cursor.fetchone()["total_transactions"]

    cursor.execute(
        """
        SELECT
            u.id,
            u.name,
            u.email,
            u.is_active,
            COALESCE(pstats.current_value, 0) AS current_value,
            COALESCE(pstats.invested_value, 0) AS invested_value,
            COALESCE(tx.txn_count, 0) AS txn_count,
            COALESCE(w.wl_count, 0) AS watchlist_count
        FROM users u
        LEFT JOIN (
            SELECT
                p.user_id,
                SUM(a.quantity * a.current_price) AS current_value,
                SUM(a.quantity * a.avg_price) AS invested_value
            FROM portfolios p
            LEFT JOIN assets a ON a.portfolio_id = p.id
            GROUP BY p.user_id
        ) pstats ON pstats.user_id = u.id
        LEFT JOIN (
            SELECT p.user_id, COUNT(t.id) AS txn_count
            FROM portfolios p
            LEFT JOIN transactions t ON t.portfolio_id = p.id
            GROUP BY p.user_id
        ) tx ON tx.user_id = u.id
        LEFT JOIN (
            SELECT p.user_id, COUNT(w.id) AS wl_count
            FROM portfolios p
            LEFT JOIN watchlist w ON w.portfolio_id = p.id
            GROUP BY p.user_id
        ) w ON w.user_id = u.id
        ORDER BY u.id ASC
    """
    )
    users = cursor.fetchall()
    admin_recommendations = get_admin_recommendations(db, limit=30)
    admin_reports = get_admin_market_reports(db, limit=30)
    admin_blogs = get_blog_posts(db, published_only=False, limit=30)

    return render_template(
        "admin_dashboard.html",
        user=session["user"],
        users=users,
        total_users=total_users,
        total_transactions=total_transactions,
        admin_recommendations=admin_recommendations,
        admin_reports=admin_reports,
        admin_blogs=admin_blogs,
    )


@app.route("/stock/<symbol>")
def stock_page(symbol):
    if "user" not in session:
        return redirect("/")

    symbol = str(symbol).strip().upper()
    if not symbol:
        return redirect("/dashboard")

    asset_type = request.args.get("asset_type", "Equity")
    return render_template(
        "stock_detail.html",
        user=session["user"],
        symbol=symbol,
        asset_type=asset_type,
    )


@app.route("/payments/create-order", methods=["POST"])
def create_payment_order():
    if "user" not in session:
        return jsonify({"status": "error", "message": "Please login to continue payment."}), 401

    data = request.json or {}
    plan_code = str(data.get("plan_code", "")).strip().upper()
    plan = SUBSCRIPTION_PLAN_CATALOG.get(plan_code)
    if not plan:
        return jsonify({"status": "error", "message": "Invalid subscription plan."}), 400

    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT id FROM users WHERE lower(email)=%s LIMIT 1", (session["user"]["email"].strip().lower(),))
    user = cursor.fetchone()
    if not user:
        return jsonify({"status": "error", "message": "User not found."}), 404

    amount_paise = int(round(float(plan["price_inr"]) * 100))
    receipt = f"sub_{user['id']}_{secrets.token_hex(6)}"
    order_payload, err = create_razorpay_order(
        amount_paise=amount_paise,
        receipt=receipt,
        notes={"plan_code": plan_code, "user_email": session["user"]["email"]},
    )
    if err or not order_payload:
        return jsonify({"status": "error", "message": f"Order creation failed: {err or 'Unknown error'}"}), 500

    order_id = order_payload.get("id")
    if not order_id:
        return jsonify({"status": "error", "message": "Razorpay order response invalid."}), 500

    try:
        cursor.execute(
            """
            INSERT INTO user_subscriptions (
                user_id, plan_code, plan_name, amount, currency, duration_days, status, razorpay_order_id
            )
            VALUES (%s, %s, %s, %s, %s, %s, 'CREATED', %s)
            ON CONFLICT (razorpay_order_id) DO NOTHING
        """,
            (
                user["id"],
                plan_code,
                plan["name"],
                float(plan["price_inr"]),
                "INR",
                int(plan["duration_days"]),
                order_id,
            ),
        )
        db.commit()
    except Exception as exc:
        db.rollback()
        return jsonify({"status": "error", "message": f"Failed to save order: {str(exc)}"}), 500

    return jsonify(
        {
            "status": "success",
            "key": RAZORPAY_KEY_ID,
            "order_id": order_id,
            "amount": amount_paise,
            "currency": "INR",
            "plan_name": plan["name"],
            "plan_code": plan_code,
            "user_name": session["user"]["name"],
            "user_email": session["user"]["email"],
        }
    )


@app.route("/payments/verify", methods=["POST"])
def verify_payment():
    if "user" not in session:
        return jsonify({"status": "error", "message": "Please login first"}), 401

    data = request.json or {}
    order_id = str(data.get("razorpay_order_id", "")).strip()
    payment_id = str(data.get("razorpay_payment_id", "")).strip()
    signature = str(data.get("razorpay_signature", "")).strip()
    if not order_id or not payment_id or not signature:
        return jsonify({"status": "error", "message": "Payment verification data missing"}), 400

    if not verify_razorpay_signature(order_id, payment_id, signature):
        return jsonify({"status": "error", "message": "Signature verification failed"}), 400

    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        """
        SELECT us.id, us.user_id, us.status, us.duration_days, us.plan_name, u.email
        FROM user_subscriptions us
        JOIN users u ON u.id = us.user_id
        WHERE us.razorpay_order_id=%s
        LIMIT 1
    """,
        (order_id,),
    )
    row = cursor.fetchone()
    if not row:
        return jsonify({"status": "error", "message": "Subscription order not found"}), 404

    if str(row["email"]).strip().lower() != str(session["user"]["email"]).strip().lower():
        return jsonify({"status": "error", "message": "Order does not belong to current user"}), 403

    if row["status"] == "PAID":
        return jsonify({"status": "success", "message": "Payment already verified", "plan_name": row["plan_name"]})

    try:
        cursor.execute(
            """
            UPDATE user_subscriptions
            SET status='PAID',
                razorpay_payment_id=%s,
                razorpay_signature=%s,
                starts_at=NOW(),
                ends_at=(NOW() + make_interval(days => %s)),
                updated_at=NOW()
            WHERE id=%s
        """,
            (payment_id, signature, int(row["duration_days"]), row["id"]),
        )
        db.commit()
    except Exception as exc:
        db.rollback()
        return jsonify({"status": "error", "message": f"Could not finalize subscription: {str(exc)}"}), 500

    return jsonify({"status": "success", "message": f"Payment successful. {row['plan_name']} activated."})


@app.route("/admin/recommendations/send", methods=["POST"])
def admin_send_recommendations():
    if "user" not in session or not session["user"].get("is_admin"):
        return jsonify({"status": "error", "message": "Admin login required"}), 403

    data = request.json or {}
    symbol_input = str(data.get("symbol", "")).strip()
    symbol = resolve_market_symbol(symbol_input)
    action = str(data.get("action", "")).strip().upper()
    note = str(data.get("note", "")).strip()
    raw_user_ids = data.get("user_ids", [])

    if action not in {"BUY", "HOLD", "SELL"}:
        return jsonify({"status": "error", "message": "Action must be BUY, HOLD, or SELL"}), 400
    if not symbol:
        return jsonify({"status": "error", "message": "Symbol is required"}), 400
    if not isinstance(raw_user_ids, list) or not raw_user_ids:
        return jsonify({"status": "error", "message": "Select at least one user"}), 400

    user_ids = []
    for item in raw_user_ids:
        try:
            uid = int(item)
            if uid > 0:
                user_ids.append(uid)
        except (TypeError, ValueError):
            continue
    user_ids = sorted(set(user_ids))
    if not user_ids:
        return jsonify({"status": "error", "message": "No valid users selected"}), 400

    def parse_optional_float(value):
        if value in (None, ""):
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return "invalid"
        return parsed if parsed > 0 else "invalid"

    entry_price = parse_optional_float(data.get("entry_price"))
    target_price = parse_optional_float(data.get("target_price"))
    stop_loss = parse_optional_float(data.get("stop_loss"))
    if "invalid" in {entry_price, target_price, stop_loss}:
        return jsonify({"status": "error", "message": "Entry/Target/Stop Loss must be positive numbers"}), 400

    db = get_db()
    cursor = db.cursor()
    try:
        cursor.execute("SELECT id FROM users WHERE id = ANY(%s)", (user_ids,))
        valid_ids = sorted({int(row["id"]) for row in cursor.fetchall()})
        if not valid_ids:
            return jsonify({"status": "error", "message": "No valid users selected"}), 400

        cursor.execute(
            """
            INSERT INTO recommendations (admin_email, symbol, action, entry_price, target_price, stop_loss, note)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """,
            (
                session["user"]["email"],
                symbol,
                action,
                entry_price,
                target_price,
                stop_loss,
                note if note else None,
            ),
        )
        recommendation_id = cursor.fetchone()["id"]

        cursor.executemany(
            """
            INSERT INTO recommendation_targets (recommendation_id, user_id)
            VALUES (%s, %s)
            ON CONFLICT (recommendation_id, user_id) DO NOTHING
        """,
            [(recommendation_id, uid) for uid in valid_ids],
        )
        db.commit()
        return jsonify(
            {
                "status": "success",
                "message": f"Recommendation sent to {len(valid_ids)} users",
                "recommendation_id": recommendation_id,
            }
        )
    except Exception as exc:
        db.rollback()
        return jsonify({"status": "error", "message": f"Failed to send recommendation: {str(exc)}"}), 500


@app.route("/admin/recommendations/delete/<int:recommendation_id>", methods=["POST"])
def admin_delete_recommendation(recommendation_id):
    if "user" not in session or not session["user"].get("is_admin"):
        return jsonify({"status": "error", "message": "Admin login required"}), 403

    db = get_db()
    cursor = db.cursor()
    try:
        cursor.execute("DELETE FROM recommendations WHERE id=%s", (recommendation_id,))
        db.commit()
        if cursor.rowcount == 0:
            return jsonify({"status": "error", "message": "Recommendation not found"}), 404
        return jsonify({"status": "success", "message": "Recommendation deleted"})
    except Exception as exc:
        db.rollback()
        return jsonify({"status": "error", "message": f"Failed to delete recommendation: {str(exc)}"}), 500


@app.route("/admin/reports/send", methods=["POST"])
def admin_send_market_report():
    if "user" not in session or not session["user"].get("is_admin"):
        return jsonify({"status": "error", "message": "Admin login required"}), 403

    data = request.json or {}
    title = str(data.get("title", "")).strip()
    summary = str(data.get("summary", "")).strip()
    content = str(data.get("content", "")).strip()
    raw_user_ids = data.get("user_ids", [])

    if not title:
        return jsonify({"status": "error", "message": "Report title is required"}), 400
    if not content:
        return jsonify({"status": "error", "message": "Report content is required"}), 400
    if not isinstance(raw_user_ids, list) or not raw_user_ids:
        return jsonify({"status": "error", "message": "Select at least one user"}), 400

    user_ids = []
    for item in raw_user_ids:
        try:
            uid = int(item)
            if uid > 0:
                user_ids.append(uid)
        except (TypeError, ValueError):
            continue
    user_ids = sorted(set(user_ids))
    if not user_ids:
        return jsonify({"status": "error", "message": "No valid users selected"}), 400

    db = get_db()
    cursor = db.cursor()
    try:
        cursor.execute("SELECT id FROM users WHERE id = ANY(%s)", (user_ids,))
        valid_ids = sorted({int(row["id"]) for row in cursor.fetchall()})
        if not valid_ids:
            return jsonify({"status": "error", "message": "No valid users selected"}), 400

        cursor.execute(
            """
            INSERT INTO market_reports (admin_email, title, summary, content)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        """,
            (
                session["user"]["email"],
                title,
                summary if summary else None,
                content,
            ),
        )
        report_id = cursor.fetchone()["id"]

        cursor.executemany(
            """
            INSERT INTO market_report_targets (report_id, user_id)
            VALUES (%s, %s)
            ON CONFLICT (report_id, user_id) DO NOTHING
        """,
            [(report_id, uid) for uid in valid_ids],
        )
        db.commit()
        return jsonify(
            {
                "status": "success",
                "message": f"Market report sent to {len(valid_ids)} users",
                "report_id": report_id,
            }
        )
    except Exception as exc:
        db.rollback()
        return jsonify({"status": "error", "message": f"Failed to send market report: {str(exc)}"}), 500


@app.route("/admin/reports/delete/<int:report_id>", methods=["POST"])
def admin_delete_market_report(report_id):
    if "user" not in session or not session["user"].get("is_admin"):
        return jsonify({"status": "error", "message": "Admin login required"}), 403

    db = get_db()
    cursor = db.cursor()
    try:
        cursor.execute("DELETE FROM market_reports WHERE id=%s", (report_id,))
        db.commit()
        if cursor.rowcount == 0:
            return jsonify({"status": "error", "message": "Report not found"}), 404
        return jsonify({"status": "success", "message": "Market report deleted"})
    except Exception as exc:
        db.rollback()
        return jsonify({"status": "error", "message": f"Failed to delete report: {str(exc)}"}), 500


@app.route("/admin/blogs/create", methods=["POST"])
def admin_create_blog():
    if "user" not in session or not session["user"].get("is_admin"):
        return jsonify({"status": "error", "message": "Admin login required"}), 403

    data = request.json or {}
    title = str(data.get("title", "")).strip()
    excerpt = str(data.get("excerpt", "")).strip()
    content = str(data.get("content", "")).strip()
    is_published = bool(data.get("is_published", True))

    if not title:
        return jsonify({"status": "error", "message": "Blog title is required"}), 400
    if not content:
        return jsonify({"status": "error", "message": "Blog content is required"}), 400

    db = get_db()
    cursor = db.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO blog_posts (author_email, title, excerpt, content, is_published)
            VALUES (%s, %s, %s, %s, %s)
        """,
            (
                session["user"]["email"],
                title,
                excerpt if excerpt else None,
                content,
                is_published,
            ),
        )
        db.commit()
        return jsonify({"status": "success", "message": "Blog post saved"})
    except Exception as exc:
        db.rollback()
        return jsonify({"status": "error", "message": f"Failed to save blog: {str(exc)}"}), 500


@app.route("/admin/blogs/<int:blog_id>/toggle", methods=["POST"])
def admin_toggle_blog(blog_id):
    if "user" not in session or not session["user"].get("is_admin"):
        return jsonify({"status": "error", "message": "Admin login required"}), 403

    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT is_published FROM blog_posts WHERE id=%s LIMIT 1", (blog_id,))
    row = cursor.fetchone()
    if not row:
        return jsonify({"status": "error", "message": "Blog post not found"}), 404

    new_status = not bool(row["is_published"])
    cursor.execute("UPDATE blog_posts SET is_published=%s WHERE id=%s", (new_status, blog_id))
    db.commit()
    return jsonify({"status": "success", "message": "Blog publish status updated", "is_published": new_status})


@app.route("/admin/blogs/<int:blog_id>/delete", methods=["POST"])
def admin_delete_blog(blog_id):
    if "user" not in session or not session["user"].get("is_admin"):
        return jsonify({"status": "error", "message": "Admin login required"}), 403

    db = get_db()
    cursor = db.cursor()
    cursor.execute("DELETE FROM blog_posts WHERE id=%s", (blog_id,))
    db.commit()
    if cursor.rowcount == 0:
        return jsonify({"status": "error", "message": "Blog post not found"}), 404
    return jsonify({"status": "success", "message": "Blog post deleted"})


@app.route("/admin/users/<int:user_id>/deactivate", methods=["POST"])
def admin_deactivate_user(user_id):
    if "user" not in session or not session["user"].get("is_admin"):
        return jsonify({"status": "error", "message": "Admin login required"}), 403

    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT id, email, is_active FROM users WHERE id=%s LIMIT 1", (user_id,))
    user_row = cursor.fetchone()
    if not user_row:
        return jsonify({"status": "error", "message": "User not found"}), 404

    target_email = str(user_row["email"]).strip().lower()
    current_admin_email = str(session["user"]["email"]).strip().lower()
    if target_email == current_admin_email:
        return jsonify({"status": "error", "message": "You cannot deactivate your own account"}), 400
    if is_admin_email(target_email):
        return jsonify({"status": "error", "message": "Cannot deactivate another admin account from here"}), 400

    cursor.execute("UPDATE users SET is_active=FALSE WHERE id=%s", (user_id,))
    db.commit()
    return jsonify({"status": "success", "message": "User deactivated"})


@app.route("/admin/users/<int:user_id>/activate", methods=["POST"])
def admin_activate_user(user_id):
    if "user" not in session or not session["user"].get("is_admin"):
        return jsonify({"status": "error", "message": "Admin login required"}), 403

    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT id FROM users WHERE id=%s LIMIT 1", (user_id,))
    user_row = cursor.fetchone()
    if not user_row:
        return jsonify({"status": "error", "message": "User not found"}), 404

    cursor.execute("UPDATE users SET is_active=TRUE WHERE id=%s", (user_id,))
    db.commit()
    return jsonify({"status": "success", "message": "User activated"})


@app.route("/admin/users/<int:user_id>/delete", methods=["POST"])
def admin_delete_user(user_id):
    if "user" not in session or not session["user"].get("is_admin"):
        return jsonify({"status": "error", "message": "Admin login required"}), 403

    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT id, email FROM users WHERE id=%s LIMIT 1", (user_id,))
    user_row = cursor.fetchone()
    if not user_row:
        return jsonify({"status": "error", "message": "User not found"}), 404

    target_email = str(user_row["email"]).strip().lower()
    current_admin_email = str(session["user"]["email"]).strip().lower()
    if target_email == current_admin_email:
        return jsonify({"status": "error", "message": "You cannot delete your own account"}), 400
    if is_admin_email(target_email):
        return jsonify({"status": "error", "message": "Cannot delete another admin account from here"}), 400

    cursor.execute("DELETE FROM users WHERE id=%s", (user_id,))
    db.commit()
    return jsonify({"status": "success", "message": "User deleted"})


@app.route("/profile/update", methods=["POST"])
def profile_update():
    if "user" not in session:
        return jsonify({"status": "error", "message": "Please login first"}), 401

    data = request.json or {}
    phone = str(data.get("phone", "")).strip()
    if len(phone) > 30:
        return jsonify({"status": "error", "message": "Phone number is too long"}), 400

    allowed_chars = set("0123456789+-() ")
    if phone and any(ch not in allowed_chars for ch in phone):
        return jsonify({"status": "error", "message": "Phone number contains invalid characters"}), 400

    db = get_db()
    cursor = db.cursor()
    try:
        cursor.execute(
            "UPDATE users SET phone=%s WHERE lower(email)=%s",
            (phone if phone else None, str(session['user']['email']).strip().lower()),
        )
        db.commit()
        return jsonify({"status": "success", "message": "Profile updated successfully"})
    except Exception as exc:
        db.rollback()
        return jsonify({"status": "error", "message": f"Failed to update profile: {str(exc)}"}), 500


@app.route("/profile/change-password", methods=["POST"])
def profile_change_password():
    if "user" not in session:
        return jsonify({"status": "error", "message": "Please login first"}), 401

    data = request.json or {}
    current_password = str(data.get("current_password", "")).strip()
    new_password = str(data.get("new_password", "")).strip()
    confirm_password = str(data.get("confirm_password", "")).strip()

    if not current_password or not new_password or not confirm_password:
        return jsonify({"status": "error", "message": "All password fields are required"}), 400
    if new_password != confirm_password:
        return jsonify({"status": "error", "message": "New password and confirm password do not match"}), 400
    if len(new_password) < 4:
        return jsonify({"status": "error", "message": "New password must be at least 4 characters"}), 400

    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT id, password FROM users WHERE lower(email)=%s LIMIT 1", (str(session["user"]["email"]).strip().lower(),))
    user_row = cursor.fetchone()
    if not user_row:
        return jsonify({"status": "error", "message": "User not found"}), 404
    if not check_password_hash(user_row["password"], current_password):
        return jsonify({"status": "error", "message": "Current password is incorrect"}), 400

    try:
        new_hash = generate_password_hash(new_password)
        cursor.execute("UPDATE users SET password=%s WHERE id=%s", (new_hash, user_row["id"]))
        db.commit()
        return jsonify({"status": "success", "message": "Password changed successfully"})
    except Exception as exc:
        db.rollback()
        return jsonify({"status": "error", "message": f"Failed to change password: {str(exc)}"}), 500


@app.route("/watchlist/add", methods=["POST"])
def add_watchlist_item():
    if "user" not in session:
        return jsonify({"status": "error", "message": "Please login first"}), 401

    data = request.json or {}
    symbol = str(data.get("symbol", "")).strip().upper()
    asset_type = str(data.get("asset_type", "Equity")).strip() or "Equity"
    note = str(data.get("note", "")).strip()

    target_price_raw = data.get("target_price")
    if target_price_raw in (None, ""):
        target_price = None
    else:
        try:
            target_price = float(target_price_raw)
        except (TypeError, ValueError):
            return jsonify({"status": "error", "message": "Target price must be numeric"}), 400
        if target_price <= 0:
            return jsonify({"status": "error", "message": "Target price must be greater than 0"}), 400

    if not symbol:
        return jsonify({"status": "error", "message": "Symbol is required"}), 400

    db = get_db()
    portfolio_id = get_or_create_portfolio_id(db, session["user"]["email"])
    if not portfolio_id:
        return jsonify({"status": "error", "message": "Portfolio not found"}), 404

    cursor = db.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO watchlist (portfolio_id, symbol, asset_type, target_price, note)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (portfolio_id, symbol) DO UPDATE
            SET asset_type=EXCLUDED.asset_type,
                target_price=EXCLUDED.target_price,
                note=EXCLUDED.note
        """,
            (portfolio_id, symbol, asset_type, target_price, note if note else None),
        )
        db.commit()
        return jsonify({"status": "success", "message": f"{symbol} saved to watchlist"})
    except Exception as exc:
        db.rollback()
        return jsonify({"status": "error", "message": f"Watchlist save failed: {str(exc)}"}), 500


@app.route("/watchlist/quote/<symbol>", methods=["GET"])
def watchlist_quote(symbol):
    if "user" not in session:
        return jsonify({"status": "error", "message": "Please login first"}), 401

    symbol = str(symbol).strip().upper()
    if not symbol:
        return jsonify({"status": "error", "message": "Symbol is required"}), 400

    snapshot = get_symbol_market_snapshot(symbol)
    if not snapshot:
        return jsonify({"status": "error", "message": f"Unable to fetch quote for {symbol}. Check symbol format."}), 400

    return jsonify({"status": "success", "data": snapshot})


@app.route("/stock/chart/<symbol>", methods=["GET"])
def stock_chart(symbol):
    if "user" not in session:
        return jsonify({"status": "error", "message": "Please login first"}), 401

    symbol = str(symbol).strip().upper()
    timeframe = request.args.get("timeframe", "1D")
    if not symbol:
        return jsonify({"status": "error", "message": "Symbol is required"}), 400

    data, error_code = get_stock_chart_data(symbol, timeframe)
    if error_code == "yfinance_not_installed":
        return jsonify({"status": "error", "message": "Install yfinance to fetch chart data"}), 400
    if error_code == "no_data":
        return jsonify({"status": "error", "message": "No chart data available for this symbol/timeframe"}), 404
    if error_code:
        return jsonify({"status": "error", "message": "Failed to fetch chart data"}), 500

    return jsonify({"status": "success", "data": data})


@app.route("/stocks/suggest", methods=["GET"])
def stock_suggestions():
    if "user" not in session:
        return jsonify({"status": "error", "message": "Please login first"}), 401

    query = request.args.get("q", "")
    items = search_stock_suggestions(query)
    return jsonify({"status": "success", "items": items})


@app.route("/stock/stoploss/<symbol>", methods=["GET", "POST"])
def stock_stoploss(symbol):
    if "user" not in session:
        return jsonify({"status": "error", "message": "Please login first"}), 401

    symbol = str(symbol).strip().upper()
    if not symbol:
        return jsonify({"status": "error", "message": "Symbol is required"}), 400

    db = get_db()
    portfolio_id = get_or_create_portfolio_id(db, session["user"]["email"])
    if not portfolio_id:
        return jsonify({"status": "error", "message": "Portfolio not found"}), 404

    cursor = db.cursor()

    if request.method == "GET":
        cursor.execute(
            "SELECT stop_loss, is_triggered, triggered_price, triggered_at FROM stop_losses WHERE portfolio_id=%s AND symbol=%s LIMIT 1",
            (portfolio_id, symbol),
        )
        row = cursor.fetchone()
        value = float(row["stop_loss"]) if row else None
        if row and row.get("triggered_at") and hasattr(row["triggered_at"], "strftime"):
            triggered_at = row["triggered_at"].strftime("%Y-%m-%d %H:%M:%S")
        else:
            triggered_at = str(row.get("triggered_at")) if row and row.get("triggered_at") else None
        return jsonify(
            {
                "status": "success",
                "symbol": symbol,
                "stop_loss": value,
                "is_triggered": bool(row.get("is_triggered")) if row else False,
                "triggered_price": float(row["triggered_price"]) if row and row.get("triggered_price") is not None else None,
                "triggered_at": triggered_at,
            }
        )

    data = request.json or {}
    try:
        stop_loss = float(data.get("stop_loss", 0))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "Stop loss must be numeric"}), 400

    if stop_loss <= 0:
        return jsonify({"status": "error", "message": "Stop loss must be greater than 0"}), 400

    cursor.execute(
        """
        INSERT INTO stop_losses (portfolio_id, symbol, stop_loss)
        VALUES (%s, %s, %s)
        ON CONFLICT (portfolio_id, symbol) DO UPDATE
        SET stop_loss=EXCLUDED.stop_loss,
            is_triggered=FALSE,
            triggered_price=NULL,
            triggered_at=NULL,
            updated_at=NOW()
    """,
        (portfolio_id, symbol, stop_loss),
    )
    db.commit()
    return jsonify({"status": "success", "message": f"Stop loss set at {round(stop_loss, 2)}", "stop_loss": round(stop_loss, 2)})


@app.route("/market/indices", methods=["GET"])
def market_indices():
    if "user" not in session:
        return jsonify({"status": "error", "message": "Please login first"}), 401

    items, error_code = get_market_indices_snapshot()
    if error_code == "yfinance_not_installed":
        return jsonify({"status": "error", "message": "Install yfinance to fetch live indices"}), 400

    return jsonify({"status": "success", "items": items})


@app.route("/market/overview", methods=["GET"])
def market_overview():
    if "user" not in session:
        return jsonify({"status": "error", "message": "Please login first"}), 401

    payload, error_code = get_market_overview_snapshot()
    if error_code == "yfinance_not_installed":
        return jsonify({"status": "error", "message": "Install yfinance to fetch market overview"}), 400
    if not payload:
        return jsonify({"status": "error", "message": "Unable to fetch market overview"}), 500

    return jsonify({"status": "success", **payload})


@app.route("/dashboard/insights", methods=["GET"])
def dashboard_insights():
    if "user" not in session:
        return jsonify({"status": "error", "message": "Please login first"}), 401

    db = get_db()
    portfolio_id = get_or_create_portfolio_id(db, session["user"]["email"])
    if not portfolio_id:
        return jsonify({"status": "error", "message": "Portfolio not found"}), 404

    watch_news, trending_news, economic_events, watch_symbols, insights_error = get_watchlist_news_and_calendar(db, portfolio_id)
    payload = {
        "status": "success",
        "news": watch_news,
        "trending_news": trending_news,
        "events": economic_events,
        "symbols": watch_symbols,
    }
    if insights_error == "yfinance_not_installed":
        payload["warning"] = "Install yfinance to fetch live news and earnings."
    return jsonify(payload)


@app.route("/watchlist/remove/<int:item_id>", methods=["POST"])
def remove_watchlist_item(item_id):
    if "user" not in session:
        return jsonify({"status": "error", "message": "Please login first"}), 401

    db = get_db()
    portfolio_id = get_or_create_portfolio_id(db, session["user"]["email"])
    if not portfolio_id:
        return jsonify({"status": "error", "message": "Portfolio not found"}), 404

    cursor = db.cursor()
    cursor.execute("DELETE FROM watchlist WHERE id=%s AND portfolio_id=%s", (item_id, portfolio_id))
    db.commit()

    if cursor.rowcount == 0:
        return jsonify({"status": "error", "message": "Watchlist item not found"}), 404

    return jsonify({"status": "success", "message": "Watchlist item removed"})


@app.route("/order/place", methods=["POST"])
def place_order():
    if "user" not in session:
        return jsonify({"status": "error", "message": "Please login first"}), 401

    data = request.json or {}
    symbol = str(data.get("symbol", "")).strip().upper()
    side = str(data.get("side", "")).strip().upper()
    order_type = str(data.get("order_type", "MARKET")).strip().upper()
    asset_type = str(data.get("asset_type", "Equity")).strip() or "Equity"

    try:
        quantity = float(data.get("quantity", 0))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "Quantity must be numeric"}), 400

    if side not in {"BUY", "SELL"}:
        return jsonify({"status": "error", "message": "Side must be BUY or SELL"}), 400
    if order_type not in {"MARKET", "LIMIT"}:
        return jsonify({"status": "error", "message": "Order type must be MARKET or LIMIT"}), 400
    if not symbol or quantity <= 0:
        return jsonify({"status": "error", "message": "Valid symbol and quantity are required"}), 400

    exec_price = None
    if order_type == "LIMIT":
        try:
            exec_price = float(data.get("limit_price", 0))
        except (TypeError, ValueError):
            return jsonify({"status": "error", "message": "Limit price must be numeric"}), 400
        if exec_price <= 0:
            return jsonify({"status": "error", "message": "Limit price must be greater than 0"}), 400
    else:
        snapshot = get_symbol_market_snapshot(symbol)
        if not snapshot or not snapshot.get("current_price"):
            return jsonify({"status": "error", "message": "Unable to fetch current market price"}), 400
        exec_price = float(snapshot["current_price"])

    db = get_db()
    portfolio_id = get_or_create_portfolio_id(db, session["user"]["email"])
    if not portfolio_id:
        return jsonify({"status": "error", "message": "Portfolio not found"}), 404

    cursor = db.cursor()
    try:
        cursor.execute(
            """
            SELECT id, quantity, avg_price, asset_type
            FROM assets
            WHERE portfolio_id=%s AND symbol=%s
            LIMIT 1
        """,
            (portfolio_id, symbol),
        )
        asset = cursor.fetchone()

        if side == "BUY":
            if asset:
                old_qty = float(asset["quantity"])
                old_avg = float(asset["avg_price"])
                new_qty = old_qty + quantity
                new_avg = ((old_qty * old_avg) + (quantity * exec_price)) / new_qty
                cursor.execute(
                    """
                    UPDATE assets
                    SET quantity=%s, avg_price=%s, current_price=%s, asset_type=%s
                    WHERE id=%s
                """,
                    (new_qty, round(new_avg, 2), exec_price, asset_type, asset["id"]),
                )
                asset_id = asset["id"]
            else:
                cursor.execute(
                    """
                    INSERT INTO assets (portfolio_id, symbol, asset_type, quantity, avg_price, current_price)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """,
                    (portfolio_id, symbol, asset_type, quantity, exec_price, exec_price),
                )
                asset_id = cursor.fetchone()["id"]
        else:
            if not asset:
                return jsonify({"status": "error", "message": "Asset not found in holdings for SELL order"}), 400

            current_qty = float(asset["quantity"])
            if quantity > current_qty:
                return jsonify({"status": "error", "message": "Insufficient shares for SELL order"}), 400

            new_qty = current_qty - quantity
            cursor.execute(
                """
                UPDATE assets
                SET quantity=%s, current_price=%s
                WHERE id=%s
            """,
                (new_qty, exec_price, asset["id"]),
            )
            asset_id = asset["id"]

        amount = round(quantity * exec_price, 2)
        cursor.execute(
            """
            INSERT INTO transactions (portfolio_id, asset_id, txn_type, quantity, price, amount)
            VALUES (%s, %s, %s, %s, %s, %s)
        """,
            (portfolio_id, asset_id, side, quantity, exec_price, amount),
        )

        db.commit()
        return jsonify(
            {
                "status": "success",
                "message": f"{side} order executed for {quantity} {symbol} at {round(exec_price, 2)}",
            }
        )
    except Exception as exc:
        db.rollback()
        return jsonify({"status": "error", "message": f"Order failed: {str(exc)}"}), 500


@app.route("/prices/sync", methods=["POST"])
def sync_prices():
    if "user" not in session:
        return jsonify({"status": "error", "message": "Please login first"}), 401

    db = get_db()
    portfolio_id = get_or_create_portfolio_id(db, session["user"]["email"])
    if not portfolio_id:
        return jsonify({"status": "error", "message": "Portfolio not found"}), 404

    data = request.json or {}
    mode = str(data.get("mode", "live")).lower()
    selected_symbols = [str(s).strip().upper() for s in data.get("symbols", []) if str(s).strip()]

    cursor = db.cursor()
    cursor.execute(
        "SELECT symbol, current_price, avg_price FROM assets WHERE portfolio_id=%s AND quantity > 0",
        (portfolio_id,),
    )
    rows = cursor.fetchall()

    if not rows:
        return jsonify({"status": "error", "message": "No holdings to sync"}), 400

    symbols = [row["symbol"] for row in rows]
    if selected_symbols:
        symbols = [s for s in symbols if s in set(selected_symbols)]

    if not symbols:
        return jsonify({"status": "error", "message": "No matching symbols found"}), 400

    symbol_to_row = {row["symbol"]: row for row in rows}

    if mode == "mock":
        prices = {}
        for symbol in symbols:
            base_price = float(symbol_to_row[symbol]["current_price"] or symbol_to_row[symbol]["avg_price"] or 0)
            if base_price <= 0:
                base_price = 100.0
            prices[symbol] = round(base_price * (1 + random.uniform(-0.03, 0.03)), 2)
        failed = []
        info = "mock_prices"
    else:
        prices, failed, error_code = fetch_live_prices(symbols)
        info = "live_prices"
        if error_code == "yfinance_not_installed":
            return jsonify(
                {
                    "status": "error",
                    "message": "Live sync needs yfinance package. Install with: pip install yfinance",
                }
            ), 400

    updated_count = 0
    for symbol, price in prices.items():
        cursor.execute(
            "UPDATE assets SET current_price=%s, updated_at=NOW() WHERE portfolio_id=%s AND symbol=%s",
            (price, portfolio_id, symbol),
        )
        updated_count += cursor.rowcount

    stop_loss_alerts = process_stop_loss_triggers(db, portfolio_id, session["user"]["email"])
    db.commit()

    msg = f"Updated {updated_count} assets using {info}"
    if stop_loss_alerts:
        msg += f" | Stop loss triggered for {len(stop_loss_alerts)} stock(s)"

    return jsonify(
        {
            "status": "success",
            "message": msg,
            "updated": prices,
            "failed": failed,
            "stop_loss_alerts": stop_loss_alerts,
        }
    )


@app.route("/transactions/history", methods=["GET"])
def transactions_history():
    if "user" not in session:
        return jsonify({"status": "error", "message": "Please login first"}), 401

    db = get_db()
    portfolio_id = get_or_create_portfolio_id(db, session["user"]["email"])
    if not portfolio_id:
        return jsonify({"status": "error", "message": "Portfolio not found"}), 404

    txn_type = request.args.get("txn_type")
    symbol = request.args.get("symbol")
    date_from = request.args.get("date_from")
    date_to = request.args.get("date_to")
    limit = request.args.get("limit", 20)

    try:
        rows = get_transaction_history(db, portfolio_id, txn_type, symbol, date_from, date_to, limit)
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid limit value"}), 400

    return jsonify({"status": "success", "transactions": rows})


@app.route("/rebalance/suggestions", methods=["POST"])
def rebalance_suggestions():
    if "user" not in session:
        return jsonify({"status": "error", "message": "Please login first"}), 401

    data = request.json or {}
    target = data.get(
        "target_allocations",
        {
            "Equity": 60,
            "Debt": 25,
            "Gold": 10,
            "Crypto": 5,
        },
    )

    supported = ["Equity", "Debt", "Gold", "Crypto"]
    target_clean = {}
    try:
        for asset_type in supported:
            target_clean[asset_type] = float(target.get(asset_type, 0))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "Target allocation values must be numeric"}), 400

    total_target = sum(target_clean.values())
    if abs(total_target - 100.0) > 0.5:
        return jsonify({"status": "error", "message": "Target allocation must total 100%"}), 400

    db = get_db()
    portfolio_id = get_or_create_portfolio_id(db, session["user"]["email"])
    if not portfolio_id:
        return jsonify({"status": "error", "message": "Portfolio not found"}), 404

    portfolio, holdings, type_values, type_allocations = get_portfolio_snapshot(db, portfolio_id)
    total_value = float(portfolio["current"])

    if total_value <= 0:
        return jsonify({"status": "error", "message": "No holdings available for rebalance"}), 400

    suggestions = []
    for asset_type in supported:
        current_value = float(type_values.get(asset_type, 0.0))
        target_value = total_value * (target_clean[asset_type] / 100.0)
        delta = round(target_value - current_value, 2)

        if delta > 1:
            action = "BUY"
            amount = delta
        elif delta < -1:
            action = "SELL"
            amount = abs(delta)
        else:
            action = "HOLD"
            amount = 0.0

        suggestions.append(
            {
                "asset_type": asset_type,
                "current_pct": round(type_allocations.get(asset_type, 0.0), 2),
                "target_pct": round(target_clean[asset_type], 2),
                "current_value": round(current_value, 2),
                "target_value": round(target_value, 2),
                "action": action,
                "amount": round(amount, 2),
            }
        )

    return jsonify(
        {
            "status": "success",
            "portfolio_value": round(total_value, 2),
            "suggestions": suggestions,
            "holdings_count": len(holdings),
        }
    )


@app.route("/transaction/buy", methods=["POST"])
def buy_transaction():
    if "user" not in session:
        return jsonify({"status": "error", "message": "Please login first"}), 401

    data = request.json or {}
    symbol = str(data.get("symbol", "")).strip().upper()
    asset_type = str(data.get("asset_type", "Equity")).strip() or "Equity"

    try:
        quantity = float(data.get("quantity", 0))
        price = float(data.get("price", 0))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "Quantity and price must be numbers"}), 400

    if not symbol or quantity <= 0 or price <= 0:
        return jsonify({"status": "error", "message": "Symbol, quantity and price are required"}), 400

    db = get_db()
    portfolio_id = get_or_create_portfolio_id(db, session["user"]["email"])
    if not portfolio_id:
        return jsonify({"status": "error", "message": "Portfolio not found"}), 404

    cursor = db.cursor()
    try:
        cursor.execute(
            """
            SELECT id, quantity, avg_price
            FROM assets
            WHERE portfolio_id=%s AND symbol=%s
            LIMIT 1
        """,
            (portfolio_id, symbol),
        )
        asset = cursor.fetchone()

        if asset:
            old_qty = float(asset["quantity"])
            old_avg = float(asset["avg_price"])
            new_qty = old_qty + quantity
            new_avg = ((old_qty * old_avg) + (quantity * price)) / new_qty
            cursor.execute(
                """
                UPDATE assets
                SET quantity=%s, avg_price=%s, current_price=%s, asset_type=%s
                WHERE id=%s
            """,
                (new_qty, round(new_avg, 2), price, asset_type, asset["id"]),
            )
            asset_id = asset["id"]
        else:
            cursor.execute(
                """
                INSERT INTO assets (portfolio_id, symbol, asset_type, quantity, avg_price, current_price)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """,
                (portfolio_id, symbol, asset_type, quantity, price, price),
            )
            asset_id = cursor.fetchone()["id"]

        amount = round(quantity * price, 2)
        cursor.execute(
            """
            INSERT INTO transactions (portfolio_id, asset_id, txn_type, quantity, price, amount)
            VALUES (%s, %s, 'BUY', %s, %s, %s)
        """,
            (portfolio_id, asset_id, quantity, price, amount),
        )
        db.commit()
        return jsonify({"status": "success", "message": f"Bought {quantity} of {symbol}"})
    except Exception as exc:
        db.rollback()
        return jsonify({"status": "error", "message": f"Buy failed: {str(exc)}"}), 500


@app.route("/transaction/sell", methods=["POST"])
def sell_transaction():
    if "user" not in session:
        return jsonify({"status": "error", "message": "Please login first"}), 401

    data = request.json or {}
    symbol = str(data.get("symbol", "")).strip().upper()

    try:
        quantity = float(data.get("quantity", 0))
        price = float(data.get("price", 0))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "Quantity and price must be numbers"}), 400

    if not symbol or quantity <= 0 or price <= 0:
        return jsonify({"status": "error", "message": "Symbol, quantity and price are required"}), 400

    db = get_db()
    portfolio_id = get_or_create_portfolio_id(db, session["user"]["email"])
    if not portfolio_id:
        return jsonify({"status": "error", "message": "Portfolio not found"}), 404

    cursor = db.cursor()
    try:
        cursor.execute(
            """
            SELECT id, quantity
            FROM assets
            WHERE portfolio_id=%s AND symbol=%s
            LIMIT 1
        """,
            (portfolio_id, symbol),
        )
        asset = cursor.fetchone()

        if not asset:
            return jsonify({"status": "error", "message": "Asset not found in portfolio"}), 404

        current_qty = float(asset["quantity"])
        if quantity > current_qty:
            return jsonify({"status": "error", "message": "Insufficient quantity to sell"}), 400

        new_qty = current_qty - quantity
        cursor.execute(
            """
            UPDATE assets
            SET quantity=%s, current_price=%s
            WHERE id=%s
        """,
            (new_qty, price, asset["id"]),
        )

        amount = round(quantity * price, 2)
        cursor.execute(
            """
            INSERT INTO transactions (portfolio_id, asset_id, txn_type, quantity, price, amount)
            VALUES (%s, %s, 'SELL', %s, %s, %s)
        """,
            (portfolio_id, asset["id"], quantity, price, amount),
        )

        db.commit()
        return jsonify({"status": "success", "message": f"Sold {quantity} of {symbol}"})
    except Exception as exc:
        db.rollback()
        return jsonify({"status": "error", "message": f"Sell failed: {str(exc)}"}), 500


@app.route("/logout")
def logout():
    session.clear()
    return jsonify({"message": "Logged out successfully"})


@app.route("/register", methods=["POST"])
def register():
    data = request.json or {}
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")

    hashed_password = generate_password_hash(password)
    db = get_db()
    cursor = db.cursor()

    try:
        cursor.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s) RETURNING id", (name, email, hashed_password))
        user_id = cursor.fetchone()["id"]
        cursor.execute("INSERT INTO portfolios (user_id, name) VALUES (%s, %s)", (user_id, "Primary"))
        db.commit()
        return jsonify({"status": "success", "message": "User registered successfully!"})
    except psycopg2.IntegrityError:
        db.rollback()
        return jsonify({"status": "error", "message": "Email already exists!"})


@app.route("/auth/forgot/request", methods=["POST"])
def forgot_password_request():
    data = request.json or {}
    email = str(data.get("email", "")).strip().lower()
    if not email:
        return jsonify({"status": "error", "message": "Email is required"}), 400

    db = get_db()
    cursor = db.cursor()

    # Always return generic success to avoid account enumeration
    generic_message = "If this email is registered, OTP has been sent."

    cursor.execute("SELECT id FROM users WHERE lower(email)=%s LIMIT 1", (email,))
    user = cursor.fetchone()
    if not user:
        return jsonify({"status": "success", "message": generic_message})

    cursor.execute(
        """
        SELECT 1
        FROM password_reset_otps
        WHERE lower(email)=%s
          AND created_at >= (NOW() - INTERVAL '60 seconds')
        LIMIT 1
    """,
        (email,),
    )
    if cursor.fetchone():
        return jsonify(
            {
                "status": "success",
                "message": "OTP recently sent. Please wait 60 seconds before requesting again.",
            }
        )

    otp_code = f"{secrets.randbelow(1000000):06d}"
    otp_hash = generate_password_hash(otp_code)
    expires_at = datetime.utcnow() + timedelta(minutes=10)

    try:
        cursor.execute(
            """
            INSERT INTO password_reset_otps (email, otp_hash, expires_at, used)
            VALUES (%s, %s, (NOW() + INTERVAL '10 minutes'), FALSE)
            RETURNING id
        """,
            (email, otp_hash),
        )
        otp_id = cursor.fetchone()["id"]
        db.commit()

        email_ok, email_info = send_password_reset_otp_email(email, otp_code)
        if not email_ok:
            cursor.execute("UPDATE password_reset_otps SET used=TRUE WHERE id=%s", (otp_id,))
            db.commit()
            print(f"[FORGOT OTP EMAIL ERROR] {email_info}")
            return jsonify({"status": "error", "message": "Unable to send OTP email right now."}), 500

        return jsonify({"status": "success", "message": generic_message})
    except Exception as exc:
        db.rollback()
        return jsonify({"status": "error", "message": f"Could not process request: {str(exc)}"}), 500


@app.route("/auth/forgot/reset", methods=["POST"])
def forgot_password_reset():
    data = request.json or {}
    email = str(data.get("email", "")).strip().lower()
    otp_code = str(data.get("otp", "")).strip()
    new_password = str(data.get("new_password", ""))

    if not email or not otp_code or not new_password:
        return jsonify({"status": "error", "message": "Email, OTP, and new password are required"}), 400
    if len(new_password) < 4:
        return jsonify({"status": "error", "message": "New password must be at least 4 characters"}), 400

    db = get_db()
    cursor = db.cursor()
    try:
        cursor.execute(
            """
            SELECT id, otp_hash
            FROM password_reset_otps
            WHERE lower(email)=%s
              AND used=FALSE
              AND expires_at >= NOW()
            ORDER BY created_at DESC
            LIMIT 5
        """,
            (email,),
        )
        otp_rows = cursor.fetchall()
        otp_row = None
        for row in otp_rows:
            if check_password_hash(row["otp_hash"], otp_code):
                otp_row = row
                break

        if not otp_row:
            return jsonify({"status": "error", "message": "Invalid or expired OTP"}), 400

        cursor.execute("SELECT id FROM users WHERE lower(email)=%s LIMIT 1", (email,))
        user = cursor.fetchone()
        if not user:
            return jsonify({"status": "error", "message": "Invalid request"}), 400

        new_hash = generate_password_hash(new_password)
        cursor.execute("UPDATE users SET password=%s WHERE id=%s", (new_hash, user["id"]))
        cursor.execute("UPDATE password_reset_otps SET used=TRUE WHERE id=%s", (otp_row["id"],))
        cursor.execute("UPDATE password_reset_otps SET used=TRUE WHERE lower(email)=%s AND used=FALSE", (email,))
        db.commit()
        return jsonify({"status": "success", "message": "Password reset successful. Please login with new password."})
    except Exception as exc:
        db.rollback()
        return jsonify({"status": "error", "message": f"Password reset failed: {str(exc)}"}), 500


@app.route("/contact", methods=["POST"])
def contact():
    data = request.json or {}
    name = data.get("name")
    email = data.get("email")
    message = data.get("message")

    db = get_db()
    cursor = db.cursor()
    cursor.execute("INSERT INTO contacts (name, email, message) VALUES (%s, %s, %s)", (name, email, message))
    db.commit()
    email_ok, email_info = send_contact_email(name, email, message)

    if email_ok:
        return jsonify({"status": "success", "message": "Message sent successfully!"})

    print(f"[CONTACT EMAIL ERROR] {email_info}")
    return jsonify(
        {
            "status": "success",
            "message": "Message saved, but email could not be sent right now.",
            "email_error": email_info,
        }
    )


if __name__ == "__main__":
    init_db()
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode)







