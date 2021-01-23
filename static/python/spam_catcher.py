import os
import pickle
import logging
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = "static/data/data.csv"
MODEL_PATH = "static/data/model.pkl"
N_TOP_FEATURES = 10   # Number of features to return for 'inspect' endpoint

logging.getLogger().setLevel(logging.DEBUG)


class SpamCatcher:
    """Methods for training a ham-spam classifier"""

    def __init__(self):
        self.tfidf_vectorizer = None
        self.model = None
        self.accuracy = None
        self.top_features = None

    def set_model(self,
                  save_on_new: bool = True) -> None:
        """
        | Set self.model. Uses existing model at MODEL_PATH if one exists,
        | otherwise calls self.load_and_train. Model saved to MODEL_PATH
        | if save_on_new is True.
        |
        | ---------------------------------------------------------------
        | Parameters
        | ----------
        |  save_on_new : bool
        |    If self.load_and_train invoked, whether the new model should
        |    be saved to MODEL_PATH
        |
        |
        | Returns
        | -------
        |  None
        """
        if os.path.isfile(MODEL_PATH):
            logging.debug(f"Using existing model at {MODEL_PATH}")
            with open(MODEL_PATH, "rb") as input_file:
                self.model = pickle.load(input_file)

        else:
            logging.debug(f"No model at {MODEL_PATH}; training new model")
            self.load_and_train()

            if save_on_new:
                logging.debug(f"Saving new model to {MODEL_PATH}")
                with open(MODEL_PATH, "wb") as output_file:
                    pickle.dump(self.model, output_file)

        return None

    def load_and_train(self) -> None:
        """
        | Main method for class. Instantiates self.model with random forest
        | classifier trained on data at DATA_PATH.
        """
        raw_df = pd.read_csv(DATA_PATH)

        logging.debug("Extracting features")
        clean_df = self.extract_features(raw_df['label'], raw_df['text'])

        logging.debug("Training model")
        self.train_model(clean_df)
        logging.debug("Model training complete")

        return None

    def extract_features(self,
                         labels: pd.Series,
                         docs: pd.Series) -> pd.DataFrame:
        """
        | Create dataframe where each row is a document and each column
        | is a term, weighted by TF-IDF (term frequency - inverse document
        | frequency). Lowercases all words, performs lemmatization,
        | and removes stopwords and punctuation.
        |
        | ----------------------------------------------------------------
        | Parameters
        | ----------
        |  labels : pd.Series
        |    Ham/spam classification
        |
        |  docs : pd.Series
        |    Documents to extract features from
        |
        |
        | Returns
        | -------
        |  pd.DataFrame
        """
        vectorizer = TfidfVectorizer(use_idf=True)

        # TODO: this should be just "fit", then we transform later. Save the "fit" object to self
        self.tfidf_vectorizer = None
        vectors = vectorizer.fit_transform(docs)

        # Reshape and add back ham/spam label
        feature_df = pd.DataFrame(vectors.todense(),
                                  columns=vectorizer.get_feature_names())
        feature_df.insert(0, 'label', labels)

        return feature_df

    def train_model(self,
                    df: pd.DataFrame) -> None:
        """Assumes labels are in the first column"""

        X = df.iloc[:, 1:]
        y = df.iloc[:, 0]

        y.replace({'ham': 0, 'spam': 1}, inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X_train, y_train)

        self.model = rf
        self.accuracy = round(accuracy_score(rf.predict(X_test), y_test), 4)
        self.top_features = self._get_top_features(list(X_train))
        return None

    def _get_top_features(self,
                          features: list) -> list:
        """
        | Return features sorted by importances from self.model. Number
        | limited to N_TOP_FEATURES.
        |
        | -------------------------------------------------------------
        | Parameters
        | ----------
        |  features : list
        |    List of feature names from X_train
        |
        |
        | Returns
        | -------
        |  list
        |    list of tuples in format (term, weight)
        """
        tuple_list = [*zip(features, self.model.feature_importances_.round(4))]
        sorted_list = sorted(tuple_list, key=lambda x: x[1], reverse=True)

        return sorted_list[:N_TOP_FEATURES]
