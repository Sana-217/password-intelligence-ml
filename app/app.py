import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.password_generator import generate_secure_memorable_password
from flask import Flask, render_template, request, redirect, session

app = Flask(__name__)
app.secret_key = "super_secret_key_change_this"

# Temporary in-memory storage (we'll improve later)
users = {}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        users[username] = password
        return redirect("/login")

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username in users and users[username] == password:
            session["user"] = username
            return redirect("/dashboard")

        return "Invalid credentials"

    return render_template("login.html")

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():

    if request.method == "POST":
        phrase = request.form.get("phrase")

        pwd, mem, strength = generate_secure_memorable_password(
            mode="phrase",
            phrase=phrase
        )

        # Convert memorability
        mem_label = "Easy" if mem == 1 else "Hard"

        # Convert strength
        if strength == 0:
            strength_label = "Weak"
        elif strength == 1:
            strength_label = "Medium"
        else:
            strength_label = "Strong"

        return render_template(
            "dashboard.html",
            password=pwd,
            memorability=mem_label,
            strength=strength_label
        )

    # 🔹 IMPORTANT: handle GET request
    return render_template("dashboard.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)