from sentence_transformers import CrossEncoder
import spacy
import torch

from src.paths import MODELS_DIR


class CrossEncoderIR:
    def __init__(self, model_name=MODELS_DIR / "ms-marco-MiniLM-L-6-v2", max_sentences=10, max_len=128, score_threshold=0, ignore_threshold=False, device="cuda"):
        self.model_name = model_name
        self.max_sentences = max_sentences
        self.max_len = max_len
        self.model = CrossEncoder(model_name, max_length=max_len, device=device)
        self.device = device
        self.score_threshold = score_threshold
        self.batch_size = 128
        self.ignore_threshold = ignore_threshold


    def get_top_sentences(self, query, candidates, ignore_threshold=False):
        score_tuples = []
        i = 0
        while i < len(candidates):
            batch_candidates = candidates[i:i+self.batch_size]
            i += self.batch_size
            input_tuples = [(query, candidate) for candidate in batch_candidates]
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                scores = self.model.predict(input_tuples).tolist()
            score_tuples.extend(list(zip(batch_candidates, scores)))
        sorted_score_tuples = sorted(score_tuples, key=lambda x: -x[1])
        if self.max_len is None:
            return sorted_score_tuples
        if self.ignore_threshold or ignore_threshold:
            sorted_score_tuples = [score_tuple for score_tuple in sorted_score_tuples][:self.max_sentences]
        else:
            sorted_score_tuples = [score_tuple for score_tuple in sorted_score_tuples if score_tuple[1] >= self.score_threshold][:self.max_sentences]
        return sorted_score_tuples


# def is_sports_query(query):
#     pattern = re.compile(r'\b(basketball|nba|ncaa|ncaab|nba finals|playoffs|3-point|field goal|'
#                          r'free throw|double-double|triple-double|mvp|all-star|'
#                          r'soccer|football|fifa|world cup|uefa|champions league|premier league|la liga|'
#                          r'bundesliga|serie a|goal(?:s)?|assist(?:s)?|penalty(?: kick| shootout)?|'
#                          r'hat trick|red card|yellow card|offside|corner kick|free kick|'
#                          r'stadium|team|match|game|player|coach|manager|tournament|season|'
#                          r'score|win|lose|draw|fixture|league|cup|championship|trophy)\b', re.IGNORECASE)
#     return bool(pattern.search(query))

# # def is_finance_query(query):
# #     pattern = re.compile(r'\b(stock(?:s| price(?:s)?| trading| performance| activity)?|'
# #                          r'(?:ceo|cfo|cio|coo) of|'
# #                          r'company|dividend(?:s)?|funding|fund(?:s| managers)?|market(?:s| cap| capitalization)?|'
# #                          r'index(?:es)?|bond(?:s| market| yields)?|share(?:s| price|holder)?|finance|financial|earnings|'
# #                          r'(?:P\/E|price-to-earnings) ratio|debt(?:s)?|debt-to-equity|corporate|Dow Jones|S&P|NASDAQ|IPO|'
# #                          r'treasury|yield(?:s)?|merger(?:s| & acquisitions| M&A)?|acquisition(?:s)?|'
# #                          r'trading (?:volume|activity|day|week|month|year)|open price|closing price|'
# #                          r'return on (?:assets|equity|investment)|financial statement(?:s)?|balance sheet|'
# #                          r'profit(?:s)?|loss(?:es)?|revenue|expenses|income|cash flow)\b', re.IGNORECASE)
# #     return bool(pattern.search(query))

# def is_finance_query(query):
#     pattern = re.compile(r'\b(stock(?:s| price(?:s)?| trading| performance| activity)?|'
#                          r'company|dividend(?:s)?|funding|fund(?:s| managers)?|market(?:s| cap| capitalization)?|'
#                          r'index(?:es)?|bond(?:s| market| yields)?|share(?:s| price|holder)?|finance|financial|earnings|'
#                          r'(?:P\/E|price-to-earnings) ratio|debt(?:s)?|debt-to-equity|corporate|Dow Jones|S&P|NASDAQ|IPO|'
#                          r'treasury|yield(?:s)?|merger(?:s| & acquisitions| M&A)?|acquisition(?:s)?|'
#                          r'trading (?:volume|activity|day|week|month|year)|open price|closing price|'
#                          r'return on (?:assets|equity|investment)|financial statement(?:s)?|balance sheet|'
#                          r'profit(?:s)?|loss(?:es)?|revenue|expenses|income|cash flow)\b', re.IGNORECASE)
#     return bool(pattern.search(query))

# def is_sports_query(query):
#     pattern = re.compile(r'\b(nba finals|playoffs|3-point|field goal|'
#                          r'free throw|double-double|triple-double|mvp|all-star|'
#                          r'uefa|champions league|premier league|la liga|'
#                          r'bundesliga|serie a|goal(?:s)?|assist(?:s)?|penalty(?: kick| shootout)?|'
#                          r'hat trick|red card|yellow card|offside|corner kick|free kick|'
#                          r'stadium|team|match|game|player|coach|manager|tournament|season|'
#                          r'score|win|lose|draw|fixture|league|cup|championship|trophy)\b', re.IGNORECASE)
#     return bool(pattern.search(query))

# misses = len([d for d in data if d["gpt_label"] == "miss"])
# incorrects = len([d for d in data if d["gpt_label"] == "incorrect"])
# corrects = 500 - misses - incorrects