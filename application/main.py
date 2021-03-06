from flask import Flask, render_template, request, jsonify
from utils.tools import *
app = Flask(__name__)

@app.route("/home", methods=['GET', 'POST'])
def home():
    return render_template("index.html")

@app.route("/app", methods=['GET', 'POST'])
def demo():
    return render_template("main.html")


@app.route("/result", methods=['GET', 'POST'])
def main():
    req = request.form
    num_passenger, drop, pick = req.get("num_passengers"), req.get("dropoff"), req.get("pickup")
    if validation(num_passenger, drop, pick):
        fare = parse(num_passenger, drop, pick)
        if fare == None:
            req = {"response": "Unable to recognize your address. Please try again"}
        else:
            req = {"response": "Your predicted fare: $" + str(fare)}
    else:
        req = {"response": "Invalid input. Please try again"}
    return render_template("result.html", result=req)


if __name__ == "__main__":
    app.run(debug=True)
