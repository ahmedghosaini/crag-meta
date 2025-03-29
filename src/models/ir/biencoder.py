from sentence_transformers import SentenceTransformer, util
import torch

from src.paths import MODELS_DIR


class BiEncoderIR:
    def __init__(self, model_name="multi-qa-MiniLM-L6-cos-v1", max_sentences=10, max_len=128, score_threshold=0, device="cuda"):
        self.model_name = model_name
        self.max_sentences = max_sentences
        self.max_len = max_len
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        self.score_threshold = score_threshold
        self.batch_size = 128


    def get_top_sentences(self, query, candidates, ignore_thresold=False):
        score_tuples = []
        query_embedding = self.model.encode(query.lower(), convert_to_tensor=True)
        candidate_embeddings = []
        i = 0
        while i < len(candidates):
            batch_candidates = candidates[i:i+self.batch_size]
            i += self.batch_size
            with torch.no_grad():
                candidate_embeddings.append(self.model.encode(batch_candidates, convert_to_tensor=True))

        scores = util.dot_score(query_embedding, torch.vstack(candidate_embeddings)).flatten().tolist()
        score_tuples = zip(candidates, scores)
        sorted_score_tuples = sorted(score_tuples, key=lambda x: -x[1])
        if self.max_len is None:
            return sorted_score_tuples
        elif ignore_thresold:
            sorted_score_tuples = sorted_score_tuples[:self.max_sentences]
        else:
            sorted_score_tuples = [score_tuple for score_tuple in sorted_score_tuples if score_tuple[1] >= self.score_threshold][:self.max_sentences]
        return sorted_score_tuples
