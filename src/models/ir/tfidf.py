from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer, TfidfVectorizer
from src.paths import MODELS_DIR
from sklearn.pipeline import Pipeline
import pickle

# DEFAULT_PARAMS = {
#     "n_features": 2**22,
#     "lowercase": True,
#     "ngram_range": (1, 1),
#     "norm": "l2",
#     "strip_accents": "ascii",
#     # "sublinear_tf": True,
# }


DEFAULT_PARAMS = {
    "max_features": 50_000_000,
    "lowercase": True,
    "max_df": 0.25,
    "min_df": 3,
    "ngram_range": (1, 2),
    "norm": None,
    "strip_accents": "ascii",
    "sublinear_tf": True,
    "token_pattern": r"(?u)\b\w+\b"
}


class TFIDF_IR:
    def __init__(self, tfidf_params=DEFAULT_PARAMS, max_sentences=10, models_dir=MODELS_DIR):
        self.max_sentences = max_sentences
        # self.model = Pipeline([('count', HashingVectorizer(**tfidf_params)), ("tfidf", TfidfTransformer(sublinear_tf=True))])
        self.model = TfidfVectorizer(**tfidf_params)
        self.models_dir = models_dir

    def fit(self, data):
        self.model.fit(data)

    def save_model(self):
        with open(self.models_dir / "tfidf.pkl", "wb") as f:
            pickle.dump(self.model, f)
    
    def load_model(self):
        with open(self.models_dir / "tfidf.pkl" , "rb") as f:
            self.model = pickle.load(f)
        return self

    def get_top_sentences(self, query, candidates, ignore_thresold=False):
        query_vector = self.model.transform([query])
        candidate_vectors = self.model.transform(candidates)
        scores = (query_vector @ candidate_vectors.T).todense().tolist()[0]
        score_tuples = list(zip(candidates, scores))
        sorted_score_tuples = sorted(score_tuples, key=lambda x: -x[1])
        if self.max_sentences is None:
            return sorted_score_tuples
        return sorted_score_tuples[:self.max_sentences]


def split_wiki(string):
    combined = []
    for segment in string.split("\n\n"):
        if len(combined) > 3:
            break
        if len(segment) < 50:
            break
        combined.append(segment)
    if not combined:
        return string.split("\n\n")[0]
    return "\n".join(combined)


# if __name__ == "__main__":
    # from datasets import load_dataset 

    # dataset = load_dataset("wikimedia/wikipedia", "20231101.en")
    # dataset = [split_wiki(d["text"]) for d in dataset["train"]]
    # print(len(dataset))
    # tfidf = TFIDF_IR()
    # tfidf.fit(dataset)
    # tfidf.save_model()

    # tfidf.load_model()
    # tfidf.model[0].alternate_sign=False
    # tfidf.get_top_sentences("cars are fast", ["car", "fast", "tire"])