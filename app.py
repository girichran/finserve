from flask import Flask, render_template, request, jsonify, g, session, redirect
import pymysql
import os
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # REQUIRED for sessions
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "2004")
DB_NAME = os.getenv("DB_NAME", "finserve")

# ---------------- DATABASE ----------------
def ensure_database_exists():
    safe_db_name = DB_NAME.replace("`", "``")
    setup_conn = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        autocommit=True,
    )
    setup_cur = setup_conn.cursor()
    setup_cur.execute(f"CREATE DATABASE IF NOT EXISTS `{safe_db_name}`")
    setup_cur.close()
    setup_conn.close()


def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        ensure_database_exists()
        db = g._database = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=False,
        )
    return db


def get_db_connection():
    ensure_database_exists()
    return pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        autocommit=False,
    )

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()

def init_db():
    db = get_db_connection()
    cursor = db.cursor()
    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255),
            email VARCHAR(255) UNIQUE,
            password VARCHAR(255)
        )
    """)
    # Contacts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS contacts (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255),
            email VARCHAR(255),
            message TEXT
        )
    """)
    db.commit()
    cursor.close()
    db.close()
    print("Database initialized!")

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    # Pass session info to template
    user = None
    if "user" in session:
        user = session["user"]
    return render_template("index.html", user=user)

# ---------------- LOGIN ----------------
@app.route("/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
    user = cursor.fetchone()
    if user:
        print("DB password:", user["password"])
        print("Entered password:", password)
    else:
        print("No user found with this email")
  
    if user and check_password_hash(user["password"], password):
        session["user"] = {
        "name": user["name"],
        "email": user["email"]
        }
        return jsonify({"status": "success", "message": f"Welcome {user['name']}!"})
    else:
        return jsonify({"status": "error", "message": "Invalid email or password!"})
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

    if result and check_password_hash(result[0], password):
        return render_template('userlog.html')

    return render_template('index.html', msg='Incorrect Credentials')

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")

    # Dummy portfolio data (later replace with DB)
    portfolio = {
        "invested": 1250000,
        "current": 1420000,
        "returns": 13.6,
        "risk": "Moderate"
    }

    holdings = [
        {"asset": "HDFC Mutual Fund", "type": "Equity", "invested": 200000, "current": 240000},
        {"asset": "SBI Bond", "type": "Debt", "invested": 100000, "current": 110000},
        {"asset": "ICICI Bluechip", "type": "Equity", "invested": 300000, "current": 360000}
    ]

    return render_template(
        "dashboard.html",
        user=session["user"],
        portfolio=portfolio,
        holdings=holdings
    )
@app.route("/logout")
def logout():
    session.clear()
    return jsonify({"message": "Logged out successfully"})


# ---------------- REGISTER ----------------
@app.route("/register", methods=["POST"])
def register():
    data = request.json
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")
    hashed_password = generate_password_hash(password)
    db = get_db()
    cursor = db.cursor()
    try:
        cursor.execute("INSERT INTO users (name,email,password) VALUES (%s, %s, %s)", (name, email, hashed_password))
        db.commit()
        return jsonify({"status": "success", "message": "User registered successfully!"})
    except pymysql.err.IntegrityError:
        return jsonify({"status": "error", "message": "Email already exists!"})



# ---------------- CONTACT ----------------
@app.route("/contact", methods=["POST"])
def contact():
    data = request.json
    name = data.get("name")
    email = data.get("email")
    message = data.get("message")

    db = get_db()
    cursor = db.cursor()
    cursor.execute("INSERT INTO contacts (name,email,message) VALUES (%s, %s, %s)", (name, email, message))
    db.commit()

    return jsonify({"status": "success", "message": "Message saved successfully!"})

# ---------------- MAIN ----------------
if __name__ == "__main__":
    init_db()



