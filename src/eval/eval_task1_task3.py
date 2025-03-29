import argparse
import json
from pathlib import Path
import sys
from functools import partial
from tqdm import tqdm

from src.data.utils import read_jsonl, extract_text_by_headers_html, dedup_results, html2text_parser, html2text_parser_traverse
import torch
# from src.models.ir.crossencoder import CrossEncoderIR
from src.models.ir import CrossEncoderIR, BiEncoderIR, TFIDF_IR, EnsembleIR
from src.models.llm.llama import LlamaLLM
from src.models.baseline_pipeline import BaselinePipeline
# from sentence_transformers import CrossEncoder
# from src.models.ir.tfidf import TFIDF_IR
import time
import random


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=int, default=0, choices=[0, 1])
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--ir_model", type=str, choices=["tfidf", "biencoder", "crossencoder", "ensemble"])
    parser.add_argument("--prompt_version", type=str, default="v0", choices=["v0", "v1", "no_context"])
    parser.add_argument("--include_examples", type=str2bool, default=False)
    parser.add_argument("--num_top_docs", type=int, default=10)
    parser.add_argument("--use_test_set", action="store_true")
    parser.add_argument("--use_task3_test_set", action="store_true")
    parser.add_argument("--use_peft", type=str2bool, default=True)
    parser.add_argument("--peft_path", default="models/peft")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)
    preprocessing_fn = html2text_parser_traverse
    if args.ir_model == "tfidf":
        ir_model = TFIDF_IR(max_sentences=args.num_top_docs).load_model()
    elif args.ir_model == "biencoder":
        ir_model = BiEncoderIR(max_sentences=args.num_top_docs)
    elif args.ir_model == "crossencoder":
        ir_model = CrossEncoderIR(max_sentences=args.num_top_docs, score_threshold=-1)
    else:
        ir_models = [
            TFIDF_IR(max_sentences=None).load_model(),
            CrossEncoderIR(max_sentences=None, score_threshold=float("-inf")),
        ]
        ir_model = EnsembleIR(ir_models, max_sentences=args.num_top_docs)
    llama = LlamaLLM(prompt_format=args.prompt_version, include_context_examples=args.include_examples, use_peft=args.use_peft, peft_path=args.peft_path)
    pipeline = BaselinePipeline(preprocessing_fn, ir_model, llama)
    
    json_objs = read_jsonl("data/raw/task1/crag_task_1_dev_v3_release.jsonl", -1)
    if args.num_samples >= 0:
        json_objs = [json_obj for json_obj in json_objs if json_obj["split"] == args.split][:args.num_samples]
    elif args.use_test_set:
        random_seed = random.Random(0)
        random_seed.shuffle(json_objs)
        json_objs = json_objs[-500:]
    elif args.use_task3_test_set:
        with open("data/raw/task3/dev_task3_test.json") as f:
            task_3_queries = set([x["query"] for x in json.load(f)])
        json_objs = [json_obj for json_obj in json_objs if json_obj["query"] in task_3_queries]
    print("Number of samples:", len(json_objs))
    results = []
    for json_obj in tqdm(json_objs):
        answer, query, search_results, query_time = json_obj["answer"], json_obj["query"], json_obj["search_results"], json_obj["query_time"]
        deduped_search_results = dedup_results(search_results)
        candidates = []
        for i, search_result in enumerate(deduped_search_results):
            for segment in preprocessing_fn(search_result):
                candidates.append(segment)

        top_segments = ir_model.get_top_sentences(query, candidates)
        segments_text = [f"<DOC>\n{segment[0]}\n</DOC>" for segment in top_segments]
        segments_text = '\n\n'.join(segments_text).strip()
        text_representation = f"<DOCS>\n{segments_text}\n</DOCS>"

        query = json_obj["query"]
        search_results = json_obj["search_results"]
        true_answer = json_obj["answer"]
        pred_answer = pipeline.generate_answer(query, search_results, query_time)
        results.append(
            {
                "query": query,
                "query_time": query_time,
                "ground_truth": true_answer,
                "prediction": pred_answer,
                "partial_prompt": text_representation
            }
        )
        print(query)
        print(true_answer)
        print(pred_answer)
    with open(f"results/results_{args.ir_model}_{args.prompt_version}_{args.include_examples}.json", "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()