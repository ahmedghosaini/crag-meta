

class EnsembleIR:
    def __init__(self, ir_models, max_sentences=10, max_len=128, score_threshold=0, device="cuda"):
        self.ir_models = ir_models
        self.max_sentences = max_sentences
        self.max_len = max_len
        self.device = device
        self.score_threshold = score_threshold
        self.batch_size = 128


    def get_top_sentences(self, query, candidates, ignore_thresold=False):
        candidate_rank_map = {}
        for ir_model in self.ir_models:
            sorted_scored_candidates = ir_model.get_top_sentences(query, candidates)
            for rank, (candidate, _) in enumerate(sorted_scored_candidates):
                if candidate not in candidate_rank_map:
                    candidate_rank_map[candidate] = [rank]
                else:
                    candidate_rank_map[candidate].append(rank)
        ranked_tuples = [(candidate, sum(ranks) / len(ranks)) for candidate, ranks in candidate_rank_map.items()]
        sorted_ranked_tuples = sorted(ranked_tuples, key=lambda x: x[1])
        if self.max_len is None:
            return sorted_ranked_tuples
        return sorted_ranked_tuples[:self.max_sentences]
