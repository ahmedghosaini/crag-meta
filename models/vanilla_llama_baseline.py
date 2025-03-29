import os
from typing import Any, Dict, List

from openai import OpenAI


#### CONFIG PARAMETERS ---

# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
AICROWD_SUBMISSION_BATCH_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

#### CONFIG PARAMETERS END---
# os.environ['INTERWEB_HOST'] = "http://gpunode04.kbs:11434/v1/"
# os.environ['INTERWEB_APIKEY'] = "ollama"
os.environ["INTERWEB_APIKEY"] = "D40tLQc4plxanXT91P3zEJ2Dk1mjNUJOhjxT7uCKkZPgFq1NO1Ew8ZLI47KICpku"

class InstructModel:
    def __init__(self):
        self.initialize_models()

    def initialize_models(self):
        self.model_name = "llama3.3:70b"

        self.llm = OpenAI(
            base_url=os.getenv("INTERWEB_HOST", "https://interweb.l3s.uni-hannover.de"),
            api_key=os.getenv("INTERWEB_APIKEY"),
        )

    def get_batch_size(self) -> int:
        """
        Determines the batch size that is used by the evaluator when calling the `batch_generate_answer` function.

        Returns:
            int: The batch size, an integer between 1 and 16. This value indicates how many
                 queries should be processed together in a single batch. It can be dynamic
                 across different batch_generate_answer calls, or stay a static value.
        """
        self.batch_size = AICROWD_SUBMISSION_BATCH_SIZE  
        return self.batch_size

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        """
        Generates answers for a batch of queries using associated (pre-cached) search results and query times.

        Parameters:
            batch (Dict[str, Any]): A dictionary containing a batch of input queries with the following keys:
                - 'interaction_id;  (List[str]): List of interaction_ids for the associated queries
                - 'query' (List[str]): List of user queries.
                - 'search_results' (List[List[Dict]]): List of search result lists, each corresponding
                                                      to a query. Please refer to the following link for
                                                      more details about the individual search objects:
                                                      https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/master/docs/dataset.md#search-results-detail
                - 'query_time' (List[str]): List of timestamps (represented as a string), each corresponding to when a query was made.

        Returns:
            List[str]: A list of plain text responses for each query in the batch. Each response is limited to 75 tokens.
            If the generated response exceeds 75 tokens, it will be truncated to fit within this limit.

        Notes:
        - If the correct answer is uncertain, it's preferable to respond with "I don't know" to avoid
          the penalty for hallucination.
        - Response Time: Ensure that your model processes and responds to each query within 30 seconds.
          Failing to adhere to this time constraint **will** result in a timeout during evaluation.
        """
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]

        formatted_prompts = self.format_prommpts(queries, query_times)

        # Aggregate answers into List[str]
        answers = []

        for prompt in formatted_prompts:
            response = self.llm.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                n=1,  # Number of output sequences to return for each prompt.
                top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                temperature=0.1,  # randomness of the sampling
                max_tokens=50,  # Maximum number of tokens to generate per output sequence.
            )

            # print("Q:", prompt)
            # print("A:", response.choices[0].message.content)
            answers.append(response.choices[0].message.content)

        return answers

    def format_prommpts(self, queries, query_times):
        """
        Formats queries and corresponding query_times using the chat_template of the model.
            
        Parameters:
        - queries (list of str): A list of queries to be formatted into prompts.
        - query_times (list of str): A list of query_time strings corresponding to each query.
            
        """
        system_prompt = "You are provided with a question and various references. Your task is to answer the question succinctly, using the fewest words possible. If the references do not contain the necessary information to answer the question, respond with 'I don't know'."
        formatted_prompts = []

        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            user_message = ""
            user_message += f"Current Time: {query_time}\n"
            user_message += f"Question: {query}\n"

            formatted_prompts.append(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ]
            )

        return formatted_prompts
