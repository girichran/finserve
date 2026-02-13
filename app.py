from flask import Flask, render_template, request, jsonify, g, session
import sqlite3
import os
from werkzeug.security import check_password_hash
from werkzeug.security import generate_password_hash, check_password_hash



app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # REQUIRED for sessions
DATABASE = "database.db"

# ---------------- DATABASE ----------------
def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()

def init_db():
    if not os.path.exists(DATABASE):
        db = sqlite3.connect(DATABASE)
        cursor = db.cursor()
        # Users table
        cursor.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT UNIQUE,
                password TEXT
            )
        """)
        # Contacts table
        cursor.execute("""
            CREATE TABLE contacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT,
                message TEXT
            )
        """)
        db.commit()
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
    cursor.execute("SELECT * FROM users WHERE email=?", (email,))
    user = cursor.fetchone()
    if user:
        print("DB password:", user["password"])
        print("Entered password:", password)
    else:
        print("No user found with this email")
    print("DB password:", user["password"])
    print("Entered password:", password)
    if user and check_password_hash(user["password"], password):
        session["user"] = {
        "name": user["name"],
        "email": user["email"]
        }
        return jsonify({"status": "success", "message": f"Welcome {user['name']}!"})
    else:
        return jsonify({"status": "error", "message": "Invalid email or password!"})
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
        cursor.execute("INSERT INTO users (name,email,password) VALUES (?, ?, ?)", (name,email,hashed_password))
        db.commit()
        return jsonify({"status": "success", "message": "User registered successfully!"})
    except sqlite3.IntegrityError:
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
    cursor.execute("INSERT INTO contacts (name,email,message) VALUES (?, ?, ?)", (name, email, message))
    db.commit()

    return jsonify({"status": "success", "message": "Message saved successfully!"})

# ---------------- MAIN ----------------
if __name__ == "__main__":
    init_db()
    #app.run(debug=True)



