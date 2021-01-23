from flask import Flask, render_template, jsonify
from static.python import SpamCatcher

spam_catcher = SpamCatcher()
spam_catcher.set_model()

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/inspect")
def inspect():
    return jsonify({'top_features': spam_catcher.top_features,
                    'accuracy': spam_catcher.accuracy})


@app.route("/classify")
def classify():

    # Note: this'll need to TF-IDF the string from the user
    return jsonify({})


if __name__ == "__main__":
    app.run(debug=True)
