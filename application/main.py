from flask import Flask, render_template, request, jsonify
from utils.tools import *
app = Flask(__name__)

@app.route("/home", methods=['GET', 'POST'])
def home():
    return render_template("main.html")


@app.route("/result", methods=['GET', 'POST'])
def main():
    req = request.form
    num_passenger, drop, pick = req.get("num_passengers"), req.get("dropoff"), req.get("pickup")
    req = {"fare": parse(num_passenger, drop, pick)}
    return render_template("result.html", result=req)


if __name__ == "__main__":
    app.run(debug=True)
