from flask import Flask, render_template, jsonify
from static import SpamCatcher

spam_catcher = SpamCatcher()
spam_catcher.set_model(save_on_new=True)

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/inspect")
def inspect():
    return jsonify({'top_features': spam_catcher.top_features,
                    'accuracy': spam_catcher.accuracy})


@app.route("/classify/<string:text>")
def classify(text):
    print(spam_catcher.classify_string(text))
    return jsonify(spam_catcher.classify_string(text))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
