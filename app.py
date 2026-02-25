from flask import Flask, render_template, request, jsonify, g, session, redirect
import os
import random
import json
import smtplib
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from email.message import EmailMessage

import psycopg2
from psycopg2.extras import RealDictCursor
from werkzeug.security import generate_password_hash, check_password_hash

try:
    import yfinance as yf
except ImportError:
    yf = None

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "your_secret_key_here")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "2004")
DB_NAME = os.getenv("DB_NAME", "finserve")
DATABASE_URL = os.getenv("DATABASE_URL", "")
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
CONTACT_TO_EMAIL = os.getenv("CONTACT_TO_EMAIL", SMTP_USER)
CONTACT_FROM_EMAIL = os.getenv("CONTACT_FROM_EMAIL", SMTP_USER)
ADMIN_EMAILS = {
    email.strip().lower()
    for email in os.getenv("ADMIN_EMAILS", "admin@finserve.com").split(",")
    if email.strip()
}
INDEX_SYMBOLS = [
    ("NIFTY 50", "^NSEI"),
    ("BANK NIFTY", "^NSEBANK"),
    ("NIFTY IT", "^CNXIT"),
    ("NIFTY AUTO", "^CNXAUTO"),
    ("NIFTY FMCG", "^CNXFMCG"),
    ("NIFTY PHARMA", "^CNXPHARMA"),
    ("NIFTY METAL", "^CNXMETAL"),
]


def send_contact_email(name, email, message):
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASSWORD or not CONTACT_TO_EMAIL:
        return False, "SMTP is not configured"

    msg = EmailMessage()
    msg["Subject"] = f"New Contact Message from {name}"
    msg["From"] = CONTACT_FROM_EMAIL or SMTP_USER
    msg["To"] = CONTACT_TO_EMAIL
    msg["Reply-To"] = email
    msg.set_content(
        f"New contact message received.\n\nName: {name}\nEmail: {email}\n\nMessage:\n{message}\n"
    )

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
            password VARCHAR(255)
        )
    """
    )

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
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (portfolio_id) REFERENCES portfolios(id) ON DELETE CASCADE,
            UNIQUE (portfolio_id, symbol)
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


def is_admin_email(email):
    e = str(email or "").strip().lower()
    return e in ADMIN_EMAILS or e == "giricharan4321@gmail.com"


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

    returns_pct = round(((current_total - invested_total) / invested_total) * 100, 2) if invested_total > 0 else 0.0

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
    if "user" in session:
        user = session["user"]
    return render_template("index.html", user=user)


@app.route("/login", methods=["POST"])
def login():
    data = request.json or {}
    email = data.get("email")
    password = data.get("password")

    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
    user = cursor.fetchone()

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

    if not user or not check_password_hash(user["password"], password):
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
    transactions = get_transaction_history(db, portfolio_id, limit=15)
    watchlist = get_watchlist(db, portfolio_id)

    return render_template(
        "dashboard.html",
        user=session["user"],
        portfolio=portfolio,
        holdings=holdings,
        transactions=transactions,
        type_allocations=type_allocations,
        watchlist=watchlist,
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

    return render_template(
        "admin_dashboard.html",
        user=session["user"],
        users=users,
        total_users=total_users,
        total_transactions=total_transactions,
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
            "SELECT stop_loss FROM stop_losses WHERE portfolio_id=%s AND symbol=%s LIMIT 1",
            (portfolio_id, symbol),
        )
        row = cursor.fetchone()
        value = float(row["stop_loss"]) if row else None
        return jsonify({"status": "success", "symbol": symbol, "stop_loss": value})

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
        SET stop_loss=EXCLUDED.stop_loss
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
            "UPDATE assets SET current_price=%s WHERE portfolio_id=%s AND symbol=%s",
            (price, portfolio_id, symbol),
        )
        updated_count += cursor.rowcount

    db.commit()

    return jsonify(
        {
            "status": "success",
            "message": f"Updated {updated_count} assets using {info}",
            "updated": prices,
            "failed": failed,
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




