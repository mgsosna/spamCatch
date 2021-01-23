import os
import pickle
from flask import Flask, render_template, jsonify
from sklearn.ensemble import RandomForestClassifier

from static.python import ModelTrainer

app = Flask(__name__)

# Load model if available
if os.path.isfile("static/data/model.pkl"):

    with open("static/data/model.pkl", "rb") as input_file:
        model = pickle.load(input_file)

# Otherwise instantiate and save model
else:
    model = ModelTrainer()
    model.start()

    with open("static/data/model.pkl", "wb") as output_file:
        pickle.dump(model, output_file)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/inspect")
def inspect():
    return jsonify({'top_features': model.top_features,
                    'accuracy': model.accuracy})


if __name__ == "__main__":
    app.run(debug=True)
