import bz2
import json
import os
from datetime import datetime

from openai import APIConnectionError, OpenAI, RateLimitError
from tqdm.auto import tqdm


INSTRUCTIONS = """
# Task: 
You are given a Question, a model Prediction, and a list of Ground Truth answers, judge whether the model Prediction matches any answer from the list of Ground Truth answers. Follow the instructions step by step to make a judgement. 
1. If the model prediction matches any provided answers from the Ground Truth Answer list, "Accuracy" should be "True"; otherwise, "Accuracy" should be "False".
2. If the model prediction says that it couldn't answer the question or it doesn't have enough information, "Accuracy" should always be "False".
3. If the Ground Truth is "invalid question", "Accuracy" is "True" only if the model prediction is exactly "invalid question".
# Output: 
Respond with only a single JSON string with an "Accuracy" field which is "True" or "False".
"""

IN_CONTEXT_EXAMPLES = """
# Examples:
Question: how many seconds is 3 minutes 15 seconds?
Ground truth: ["195 seconds"]
Prediction: 3 minutes 15 seconds is 195 seconds.
Accuracy: True

Question: Who authored The Taming of the Shrew (published in 2002)?
Ground truth: ["William Shakespeare", "Roma Gill"]
Prediction: The author to The Taming of the Shrew is Roma Shakespeare.
Accuracy: False

Question: Who played Sheldon in Big Bang Theory?
Ground truth: ["Jim Parsons", "Iain Armitage"]
Prediction: I am sorry I don't know.
Accuracy: False
"""



def get_system_message():
    """Returns the system message containing instructions and in context examples."""
    return INSTRUCTIONS + IN_CONTEXT_EXAMPLES


def attempt_api_call(client, model_name, messages, max_retries=10):
    """Attempt an API call with retries upon encountering specific errors."""
    # todo: add default response when all efforts fail
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0
            )
            return response.choices[0].message.content
        except (APIConnectionError, RateLimitError):
            print(
                f"API call failed on attempt {attempt + 1}, retrying..."
            )
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
    return None


def log_response(messages, response, output_directory="api_responses"):
    """Save the response from the API to a file."""
    os.makedirs(output_directory, exist_ok=True)
    file_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S.json")
    file_path = os.path.join(output_directory, file_name)
    with open(file_path, "w") as f:
        json.dump({"messages": messages, "response": response}, f)


def parse_response(resp: str):
    """Pass auto-eval output from the evaluator."""
    try:
        resp = resp.lower()
        model_resp = json.loads(resp)
        answer = -1
        if "accuracy" in model_resp and (
            (model_resp["accuracy"] is True)
            or (
                isinstance(model_resp["accuracy"], str)
                and model_resp["accuracy"].lower() == "true"
            )
        ):
            answer = 1
        else:
            raise ValueError(
                f"Could not parse answer from response: {model_resp}"
            )

        return answer
    except:
        return -1


def evaluate_predictions(predictions, evaluation_model_name, openai_client):
    n_miss, n_correct, n_correct_exact = 0, 0, 0
    system_message = get_system_message()
    prediction_dicts_labeled = []
    for prediction_dict in tqdm(
        predictions, total=len(predictions), desc="Evaluating Predictions"
    ):
        query, ground_truth, prediction = (
            prediction_dict["query"],
            prediction_dict["ground_truth"],
            prediction_dict["prediction"].lower(),
        )

        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": f"Question: {query}\n Ground truth: {ground_truth}\n Prediction: {prediction}\n",
            },
        ]
        if "insufficient information" in prediction.lower():
            prediction = "i don't know"
        if prediction == "i don't know" or prediction == "i don't know.":
            n_miss += 1
            prediction_dict["gpt_label"] = "miss"
            prediction_dicts_labeled.append(prediction_dict)
            continue
        if prediction == ground_truth:
            n_correct_exact += 1
            n_correct += 1
            prediction_dict["gpt_label"] = "exact"
            prediction_dicts_labeled.append(prediction_dict)
            continue

        response = attempt_api_call(
            openai_client, evaluation_model_name, messages
        )
        if response:
            log_response(messages, response)
            eval_res = parse_response(response)
            if eval_res == 1:
                n_correct += 1
                prediction_dict["gpt_label"] = "correct"
            else:
                prediction_dict["gpt_label"] = "incorrect"
            prediction_dicts_labeled.append(prediction_dict)


    n = len(predictions)
    results = {
        "score": (2 * n_correct + n_miss) / n - 1,
        "exact_accuracy": n_correct_exact / n,
        "accuracy": n_correct / n,
        "hallucination": (n - n_correct - n_miss) / n,
        "missing": n_miss / n,
        "n_miss": n_miss,
        "n_correct": n_correct,
        "n_correct_exact": n_correct_exact,
        "total": n,
    }
    print(json.dumps(results, indent=2))
    return prediction_dicts_labeled


if __name__ == "__main__":
    import sys
    import json
    from pathlib import Path

    with open(sys.argv[1]) as f:
        predictions = json.load(f)
    
    if len(sys.argv) > 2 and sys.argv[2] == "use_test":
        import random
        random_seed = random.Random(0)
        random_seed.shuffle(predictions)
        predictions = predictions[-500:]
    EVALUATION_MODEL_NAME = os.getenv(
        "EVALUATION_MODEL_NAME", "gpt-4-0125-preview"
    )
    EVALUATION_MODEL_NAME = os.getenv(
        "EVALUATION_MODEL_NAME", "gpt-4o"
    )
    # data = []
    # with open("data/raw/task1/crag_task_1_v2.jsonl") as f:
    #     for line in f:
    #         json_obj = json.loads(line)
    #         if json_obj["split"] == 0:
    #             data.append(json_obj)

    # predictions = []
    # for d, p in zip(data, prediction_dict):
    #     predictions.append({
    #         "query": d["query"],
    #         "ground_truth": list(p.keys())[0],
    #         "prediction": list(p.values())[0],
    #     })
    # Generate predictions

    # Evaluate Predictions
    openai_client = OpenAI()
    evaluation_results = evaluate_predictions(
        predictions, EVALUATION_MODEL_NAME, openai_client
    )

    p = Path(sys.argv[1])
    output_filename = p.parent / ("labeled_" + str(p.name))
    with open(output_filename, "w") as f:
        json.dump(evaluation_results, f, indent=2)
        f.write("\n")
