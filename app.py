from flask import Flask, render_template, jsonify
from sklearn.ensemble import RandomForestClassifier

from static.python import ModelTrainer

app = Flask(__name__)

# Instantiate and train model
model = ModelTrainer().start()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/inspect")
def inspect():
    return jsonify({'top_features': model.top_features,
                    'accuracy': model.accuracy})


if __name__ == "__main__":
    app.run(debug=True)
