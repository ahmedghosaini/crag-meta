from src.data.utils import extract_text_by_headers_html, dedup_results, html2text_parser_traverse
from src.models.ir.crossencoder import CrossEncoderIR
from src.models.llm.llama import LlamaLLM
from src.models.interweb_api import ApiCall
from src.models.llm.deepseek import DeepSeek
from src.data.api_functions import *
import os
from src.paths import ROOT_DIR
import json
import multiprocessing
import os
from functools import partial
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time

# os.environ("CRAIG_MOCK_API_URL", "https://demo3.kbs.uni-hannover.de")
# CRAG_MOCK_API_URL = os.getenv("CRAG_MOCK_API_URL", "https://demo3.kbs.uni-hannover.de")
CRAG_MOCK_API_URL = os.getenv("CRAG_MOCK_API_URL", "http://localhost:8000")

# with open(ROOT_DIR / "aicrowd.json") as f:
#     AICROWD_JSON = json.load(f)
# TASK_TYPE = "TASK1"
# TASK_TYPE = "TASK2"
TASK_TYPE = "TASK3"



def pool_preprocessing_function(search_results):
    return html2text_parser_traverse(search_results)


class BaselinePipeline:
    def __init__(self, preprocessing_fn=html2text_parser_traverse, ir_model=None, llm_model=None):            
        self.preprocessing_fn = preprocessing_fn
        if ir_model is None:
            self.ir_model = CrossEncoderIR()
        else:
            self.ir_model = ir_model
        if llm_model is None:
            print("################ DeepSeek")
            # print("################ with adapters")
            self.llm_model = DeepSeek()
            # print("################ Llama")
            # self.llm_model = LlamaLLM()

        else:
            self.llm_model = llm_model
        self.batch_size = 4
        self.task_type = TASK_TYPE

    def get_batch_size(self) -> int:
        """
        Determines the batch size that is used by the evaluator when calling the `batch_generate_answer` function.

        Returns:
            int: The batch size, an integer between 1 and 16. This value indicates how many
                 queries should be processed together in a single batch. It can be dynamic
                 across different batch_generate_answer calls, or stay a static value.
        """
        return self.batch_size

    def batch_generate_answer(self, batch):
        batch_interaction_ids = batch["interaction_id"]
        batch_queries = batch["query"]
        batch_search_results = batch["search_results"]
        batch_query_times = batch["query_time"]
        # todo: make this actually batched on llm side
        results = []
        for (query, search_results, query_time) in zip(batch_queries, batch_search_results, batch_query_times):
            deduped_search_results = dedup_results(search_results)
            candidates = []
            for search_result in deduped_search_results:
                for segment in self.preprocessing_fn(search_result):
                    candidates.append(segment)
            top_segments = self.ir_model.get_top_sentences(query, candidates)
            answer = self.llm_model.process_candiates(query, query_time, top_segments)
            if "insufficient information" in answer.lower():
                answer = "i don't know"
            results.append(answer)
        return results


    def generate_answer(self, query, search_results, query_time, return_candidates=False):
        try:
            if self.task_type == "TASK1":
                print("############### task1")
                return self.generate_answer_task1(query, search_results, query_time, return_candidates)
            elif self.task_type == "TASK2":
                print("############### task2")
                return self.generate_answer_task2(query, search_results, query_time, return_candidates)
            else:
                print("############### task3")
                return self.generate_answer_task3(query, search_results, query_time, return_candidates)
        except:
            print("############### TASK COULDN'T BE IDENTIFIED")
            return "i don't know"

    def generate_answer_task1(self, query, search_results, query_time, return_candidates):
        deduped_search_results = dedup_results(search_results)
        candidates = []
        for search_result in deduped_search_results:
            for segment in self.preprocessing_fn(search_result):
                candidates.append(segment)
        top_segments = self.ir_model.get_top_sentences(query, candidates)
        segments_text = [f"<DOC>\n{segment[0]}\n</DOC>" for segment in top_segments]
        segments_text = '\n'.join(segments_text).strip()
        answer = self.llm_model.process_candiates(query, query_time, top_segments)
        print(f"############ Query: {query}")
        # print(f"# Top Segments: {top_segments}")
        print(f"############ answer: {answer}")
        print(f"############\n\n\n\n")
        if return_candidates:
            return answer, segments_text
        
        return answer    

    def generate_answer_task2(self, query, search_results, query_time, return_candidates):
        deduped_search_results = dedup_results(search_results)
        search_candidates = []
        for search_result in deduped_search_results:
            for segment in self.preprocessing_fn(search_result):
                search_candidates.append(segment)
        top_search_segments = self.ir_model.get_top_sentences(query, search_candidates)
        text_representation = f"QUERY: {query}"
        api_call = self.llm_model.process_api(text_representation)
        print("api_call:", api_call)
        if api_call == "None":
            api_docs = []
        else:
            try:
                results = ""
                if ", " in api_call:
                    for api_call in api_call.split(", "):
                        results += eval(api_call) + "\n\n"
                else:
                    results = eval(api_call)
                # print(results)
                api_docs = results.split("\n\n")
            except Exception as e:
                print(repr(e))
                results = ""
                api_docs = []
        top_api_segments = self.ir_model.get_top_sentences(query, api_docs)
        # print(top_api_segments)
        all_top_segments = sorted(top_api_segments + top_search_segments, key=lambda x: -x[1])
        segments_text = [f"<DOC>\n{segment[0]}\n</DOC>" for segment in all_top_segments]
        segments_text = '\n'.join(segments_text).strip()
        answer = self.llm_model.process_candiates(query, query_time, all_top_segments)
        if return_candidates:
            return answer, segments_text
        return answer
    

    def generate_answer_task3(self, query, search_results, query_time, return_candidates=False):
        deduped_search_results = dedup_results(search_results)
        search_candidates = []
        # for search_result in deduped_search_results:
        #     segments = self.preprocessing_fn(search_result)
        #     # print(len(segments))
        #     for segment in self.preprocessing_fn(search_result):
        #         search_candidates.append(segment)
        with multiprocessing.Pool(processes=multiprocessing.cpu_count() // 2) as pool:
            nested_search_candidates = pool.map(pool_preprocessing_function, deduped_search_results)
            for nested_search_candidate in nested_search_candidates:
                # print(len(nested_search_candidate))
                for search_candidate in nested_search_candidate:
                    search_candidates.append(search_candidate)
        top_search_segments = self.ir_model.get_top_sentences(query, search_candidates)
        # text_representation = f"QUERY: {query}"
        # api_call = self.llm_model.process_api(text_representation)
        # if api_call == "None":
        #     api_docs = []
        # else:
        #     try:
        #         results = ""
        #         if ", " in api_call:
        #             for api_call in api_call.split(", "):
        #                 results += eval(api_call) + "\n\n"
        #         else:
        #             results = eval(api_call)
        #         api_docs = results.split("\n\n")
        #     except Exception as e:
        #         print(repr(e))
        #         results = ""
        #         api_docs = []
        # top_api_segments = self.ir_model.get_top_sentences(query, api_docs)
        # all_top_segments = sorted(top_api_segments + top_search_segments, key=lambda x: -x[1])
        all_top_segments = top_search_segments
        segments_text = [f"<DOC>\n{segment[0]}\n</DOC>" for segment in all_top_segments]
        segments_text = '\n'.join(segments_text).strip()
        answer = self.llm_model.process_candiates(query, query_time, all_top_segments)
        if return_candidates:
            return answer, segments_text
        return answer


    def get_expanded_query(self, query):
        text_representation = f"QUESTION: Given the query \"{query}\", what would be a better text representation for an information retrieval system? Please only include the new query in your response, as I will use it directly for the new query."
        answer = self.llm_model.process_example(text_representation)
        return answer
