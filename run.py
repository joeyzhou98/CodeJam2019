from flask import Flask, render_template, redirect, url_for, request
app = Flask(__name__)


@app.route('/')
def root():
    return redirect(url_for('home'))


@app.route('/home')
def home():
    data = request.args.get("data")
    if data is None:
        return render_template("home.html", result="No results")
    else:
        return render_template("home.html", result=data)


app.run(port=5000)
