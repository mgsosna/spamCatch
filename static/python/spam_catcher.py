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
                obj = pickle.load(input_file)
                self.tfidf_vectorizer = obj['tfidf_vectorizer']
                self.model = obj['model']
                self.accuracy = obj['accuracy']
                self.top_features = obj['top_features']

        else:
            logging.debug(f"No model at {MODEL_PATH}; training new model")
            self.load_and_train()

            if save_on_new:
                logging.debug(f"Saving new model to {MODEL_PATH}")
                with open(MODEL_PATH, "wb") as output_file:
                    pickle.dump(vars(self), output_file)

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
        if not self.tfidf_vectorizer:
            self.set_tfidf_vectorizer(docs)

        # Transform documents into TF-IDF features
        features = self.tfidf_vectorizer.transform(docs)

        # Reshape and add back ham/spam label
        feature_df = pd.DataFrame(features.todense(),
                                  columns=self.tfidf_vectorizer.get_feature_names())
        feature_df.insert(0, 'label', labels)

        return feature_df

    def set_tfidf_vectorizer(self,
                             training_docs: pd.Series) -> None:
        """
        | Fit the TF-IDF vectorizer. Updates self.tfidf_vectorizer
        |
        | ---------------------------------------------------------
        | Parameters
        | ----------
        |  training_docs : pd.Series
        |    An iterable of strings, one per document, to use for
        |    fitting the TF-IDF vectorizer
        |
        |
        | Returns
        | -------
        |  None
        """
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_vectorizer.fit(training_docs)
        return None

    def train_model(self,
                    df: pd.DataFrame) -> None:
        """
        | Train a random forest classifier on df. Assumes first column
        | is labels and all remaining columns are features. Updates
        | self.model, self.accuracy, and self.top_features
        |
        | ------------------------------------------------------------
        | Parameters
        | ----------
        |  df : pd.DataFrame
        |    The data, where first column is labels and remaining columns
        |    are features
        |
        |
        | Returns
        | -------
        |  None
        """
        X = df.iloc[:, 1:]
        y = df.iloc[:, 0]

        # Set spam as target
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

    def classify_string(self,
                        text: str) -> float:
        """
        | Get the probability that a string is spam. Transforms the
        | string into a TF-IDF vector and then returns self.model's
        | prediction on the vector.
        |
        | ---------------------------------------------------------
        | Parameters
        | ----------
        |  text : str
        |    A raw string to be classified
        """
        if not self.tfidf_vectorizer:
            raise ValueError("Cannot generate predictions; must first "
                             " set self.tfidf_vectorizer")

        vec = self.tfidf_vectorizer.transform([text])
        return self.model.predict_proba(vec)[0][1]
