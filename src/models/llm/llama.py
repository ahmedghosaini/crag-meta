import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel, PeftConfig

from src.paths import MODELS_DIR

DEFAULT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
)

import os
from openai import OpenAI

os.environ["INTERWEB_APIKEY"] = "D40tLQc4plxanXT91P3zEJ2Dk1mjNUJOhjxT7uCKkZPgFq1NO1Ew8ZLI47KICpku"

client = OpenAI(
    base_url=os.getenv("INTERWEB_HOST", "https://interweb.l3s.uni-hannover.de/v1"),
    api_key=os.getenv("INTERWEB_APIKEY"),
)


# DEFAULT_SYSTEM_INSTRUCTIONS = """
# You are a helpful, respectful and honest question answering system. I will be providing you with questions, as well as some additional context that may be helpful in answering those questions. You should output only the necessary information to answer the question, i.e. be brief.
# You will be given additional context retrieved from relevant webpages to your question beteen "<DOCS>" and "</DOCS>". Below is a couple of examples of this:

# ### EXAMPLE ONLY
# QUESTION: how much further can bald eagles see than humans?
# <DOCS>
# <DOC> They have amazing eyesight: A human with perfect eyesight has 20/20 vision. Bald eagles can have 20/4 or 20/5 vision, meaning they can see four or five times farther than the average person.</DOC>
# <DOC> There are many metrics to compare. For example, Human eyes take up only 5% of our head, eagle eyes occupy 50%. The back of their eyes is bigger and flatter, physically allowing for a bigger picture on their retinas, which are more densely covered in light detector cells (cones). Eagles see clearly for up to 5x further than a human with perfect vision.</DOC>
# </DOCS>
# ANSWER: Up to 4 to 5 times as far as an average human.

# QUESTION: which state was joe biden born?
# <DOCS>
# <DOC> Early life: Joseph Robinette Biden Jr. was born on November 20, 1942, at St. Mary's Hospital in Scranton, Pennsylvania, to Catherine Eugenia "Jean" Biden (née Finnegan) and Joseph Robinette Biden Sr. The oldest child in a Catholic family of English, French, and Irish descent, he has a sister, Valerie, and two brothers, Francis and James.</DOC>
# </DOCS>
# ANSWER: Pennsylvania.
# ### EXAMPLE ONLY

# You must base your outputs on the information within <DOCS> for the particular answer, e.g. do not simply output one of the answers above 
# """.strip()

QA_SYSTEM_INSTRUCTIONS = """
You are a helpful, respectful and honest question answering system. I will be providing you with questions, as well as some additional context that may be helpful in answering those questions. You will be provided with additional information between the "<DOCS>" tags. Keep your answers brief, ideally less than 20 words, but a strict limit of 30 words.

If the provided information is insufficient for answering the question, simply output "Insuffient information" and only output that. If the question asserts a false premise, like "When did Eisenhower become Prime Minister?", simply output "Invalid question". Thus, if you are inclined to say something like "X never has" or "X never was" or "X never did", your output should be "Invalid question".

Final reminder, with award shows, years can be tricky. Often awards are handed out the year after the project is made. Thus for the Oscars in 2015, the awards are being given out to movies made in 2014.
""".strip()

# QA_SYSTEM_INSTRUCTIONS = """
# You are a helpful, respectful and honest question answering system. I will be providing you with questions, as well as some additional context that may be helpful in answering those questions. You will be provided with additional information between the "<DOCS>" tags. Keep your answers brief, ideally less than 20 words, but a strict limit of 30 words.

# If the provided information is insufficient for answering the question, simply output "Insuffient information" and only output that. If the question asserts a false premise, like "When did Eisenhower become Prime Minister?", simply output "Invalid question". Thus, if you are inclined to say something like "X never has" or "X never was" or "X never did", your output should be "Invalid question".

# Final reminder, with award shows, years can be tricky. Often awards are handed out the year after the project is made. Thus for the Oscars in 2015, the awards are being given out to movies made in 2014.

# If the question is asking you to report a stock price, market cap, or which team played which recently, simply say "I don't know" as you don't have enough information to answer those questions. 

# When the query asks for a date give both the day and month if you can. If it asks 'what year' then just returning the year is fine. Finally, answer all questions fully, you tend to only give partial answers to questions that require multiple people for the answer.
# """.strip()

# QA_SYSTEM_INSTRUCTIONS = """
# You are a helpful, respectful and honest question answering system. I will be providing you with questions, as well as some additional context that may be helpful in answering those questions. You will be provided with additional information between the "<DOCS>" tags. Keep your answers brief, ideally less than 20 words, but a strict limit of 30 words.

# If the provided information is insufficient for answering the question, simply output "Insuffient information" and only output that. If the question asserts a false premise, like "When did Eisenhower become Prime Minister?", simply output "Invalid question". Thus, if you are inclined to say something like "X never has" or "X never was" or "X never did", your output should be "Invalid question".

# Final reminder, with award shows, years can be tricky. Often awards are handed out the year after the project is made. Thus for the Oscars in 2015, the awards are being given out to movies made in 2014. Finally, your answer must be returned in lower case.
# """.strip()

# QA_SYSTEM_INSTRUCTIONS = """
# You are a helpful, respectful and honest question answering system. I will be providing you with questions, as well as some additional context that may be helpful in answering those questions. You will be provided with additional information between the "<DOCS>" tags. Keep your answers brief, ideally less than 20 words, but a strict limit of 30 words.

# Note, the scoring for this task means the making predictions when you are unsure you are punished. There are two alternative answers you may give other than the answer:\nI don't know: Use this when the information provided is insufficient for answering the question\nInvalid question: Use this when the question asserts a false premise, for example "When did Ronald Reagan become the Prime Minister of Russia"\n\nI
# """.strip()

QA_SYSTEM_INSTRUCTIONS_NO_CONTEXT = """
You are a helpful, respectful and honest question answering system. Your job is to answer questions that you can answer without any additional context. This includes topics you have knowledge off, like simple facts that you know, as well as reasoning and common sense questions. Please keep your answers brief, they must be less than 50 word answers.

If the question asserts a false premise, like "When did Eisenhower become Prime Minister?", simply output "Invalid question".

If you do not know the answer, simple output "I don't know".

Also for date related questions (e.g. what date did X happen) simply return "I don't know" as you have a hard time giving the correct dates.
""".strip()


API_SYSTEM_INSTRUCTIONS = """
You are a system that predicts the correct Python function to call with the associated arguments. You will be given a query and based on that query, you will be given the documentation for a set of Python functions that call a remote API and return the results. Your job is to take the query and predict the correct function call and arguments based on the query.
Your response will be piped directly into Python's `eval()`, so only include the function and arguments with no additional text, otherwise you will cause an exception in the code. If you think none of the API functions can provide relevant information, simple return "None".
""".strip()

API_DOCS = """
<DOCUMENTATION>
open_search_entity_by_name(query)
    Get details about the list of entities returned by query.
    Args:
        query (str): the query
    Returns:
        A list of entities (List[str])


movie_get_person_info(query)
    Gets person info in database through BM25.
    Args:
        query (str): person name to be searched
    Returns:
        list of top n matching entities (List[Dict[str, Any]]). Entities are ranked by BM25 score. The returned entities MAY contain the following fields:
            name (string): name of person
            id (int): unique id of person
            acted_movies (list[int]): list of movie ids in which person acted
            directed_movies (list[int]): list of movie ids in which person directed
            birthday (string): string of person's birthday, in the format of "YYYY-MM-DD"
            oscar_awards: list of oscar awards (dict), win or nominated, in which the person was the entity. The format for oscar award entity are:
                'year_ceremony' (int): year of the oscar ceremony,
                'ceremony' (int): which ceremony. for example, ceremony = 50 means the 50th oscar ceremony,
                'category' (string): category of this oscar award,
                'name' (string): name of the nominee,
                'film' (string): name of the film,
                'winner' (bool): whether the person won the award


movie_get_movie_info(query)
    Gets movie info in database through BM25.
    Args:
        query (str): movie name to be searched
    Returns:
        list of top n matching entities (List[Dict[str, Any]]). Entities are ranked by BM25 score. The returned entities MAY contain the following fields:
            title (string): title of movie
            id (int): unique id of movie
            release_date (string): string of movie's release date, in the format of "YYYY-MM-DD"
            original_title (string): original title of movie, if in another language other than english
            original_language (string): original language of movie. Example: 'en', 'fr'
            budget (int): budget of movie, in USD
            revenue (int): revenue of movie, in USD
            rating (float): rating of movie, in range [0, 10]
            genres (list[dict]): list of genres of movie. Sample genre object is {'id': 123, 'name': 'action'}
            oscar_awards: list of oscar awards (dict), win or nominated, in which the movie was the entity. The format for oscar award entity are:
                'year_ceremony' (int): year of the oscar ceremony,
                'ceremony' (int): which ceremony. for example, ceremony = 50 means the 50th oscar ceremony,
                'category' (string): category of this oscar award,
                'name' (string): name of the nominee,
                'film' (string): name of the film,
                'winner' (bool): whether the person won the award
            cast (list [dict]): list of cast members of the movie and their roles. The format of the cast member entity is:
                'name' (string): name of the cast member,
                'id' (int): unique id of the cast member,
                'character' (string): character played by the cast member in the movie,
                'gender' (string): the reported gender of the cast member. Use 2 for actor and 1 for actress,
                'order' (int): order of the cast member in the movie. For example, the actress with the lowest order is the main actress,
            crew' (list [dict]): list of crew members of the movie and their roles. The format of the crew member entity is:
                'name' (string): name of the crew member,
                'id' (int): unique id of the crew member,
                'job' (string): job of the crew member,


movie_get_year_info(query)
    Gets info of a specific year
    Args:
        query (str): string of year. Note that we only support years between 1990 and 2021
    Returns:
        An entity representing year information (Dict[str, Any]). The returned entity MAY contain the following fields:
            movie_list: list of movie ids in the year. This field can be very long to a few thousand films
            oscar_awards: list of oscar awards (dict), held in that particular year. The format for oscar award entity are:
                'year_ceremony' (int): year of the oscar ceremony,
                'ceremony' (int): which ceremony. for example, ceremony = 50 means the 50th oscar ceremony,
                'category' (string): category of this oscar award,
                'name' (string): name of the nominee,
                'film' (string): name of the film,
                'winner' (bool): whether the person won the award


finance_get_price_history_by_ticker(query)
    Return 1 year history of daily Open price, Close price, High price, Low price and trading Volume.
    Args:
        query (str): ticker_name
    Returns:
        1 year daily price histor)


finance_get_detailed_price_history_by_ticker(query)
    Return the past 5 days' history of 1 minute Open price, Close price, High price, Low price and trading Volume, starting from 09:30:00 EST to 15:59:00 EST. Note that the Open, Close, High, Low, Volume are the data for the 1 min duration. However, the Open at 9:30:00 EST may not be equal to the daily Open price, and Close at 15:59:00 EST may not be equal to the daily Close price, due to handling of the paper trade. The sum of the 1 minute Volume may not be equal to the daily Volume.
    Args:
        query (str): ticker_name
    Returns:
        Past 5 days' 1 min price history


finance_get_detailed_price_history_by_name(query)
    Return the past 5 days' history of 1 minute Open price, Close price, High price, Low price and trading Volume, starting from 09:30:00 EST to 15:59:00 EST. Note that the Open, Close, High, Low, Volume are the data for the 1 min duration. However, the Open at 9:30:00 EST may not be equal to the daily Open price, and Close at 15:59:00 EST may not be equal to the daily Close price, due to handling of the paper trade. The sum of the 1 minute Volume may not be equal to the daily Volume.
    Args:
        query (str): company_name
    Returns:
        Past 5 days' 1 min price history


finance_get_divdends_history_by_ticker(query)
    Return dividend history of a ticker.
    Args:
        query (str): ticker_name
    Returns:
        Dividend distribution history


finance_get_divdends_history_by_name(query)
    Return dividend history of a ticker searching by company name.
    Args:
        query (str): company name
    Returns:
        Dividend distribution history whose format follows the below example: {'2019-12-19 00:00:00 EST': 0.058, '2020-03-19 00:00:00 EST': 0.2, '2020-06-12 00:00:00 EST': 0.2, ... }


finance_get_market_capitalization_by_ticker(query)
    Return the market capitalization of a ticker.
    Args:
        query (str): ticker_name
    Returns:
        Market capitalization (float)


finance_get_market_capitalization_by_name(query)
    Return the market capitalization of a ticker searching by company name.
    Args:
        query (str): company_name
    Returns:
        Market capitalization (float)


finance_get_eps_by_ticker(query)
    Return earnings per share of a ticker.
    Args:
        query (str): ticker_name
    Returns:
        Earnings per share (float)


finance_get_eps_by_name(query)
    Return earnings per share of a ticker searching by company name.
    Args:
        query (str): company_name
    Returns:
        Earnings per share (float)


finance_get_pe_ratio_by_ticker(query)
    Return price-to-earnings ratio of a ticker.
    Args:
        query (str): ticker_name
    Returns:
        Price-to-earnings ratio (float)


finance_get_pe_ratio_by_name(query)
    Return price-to-earnings ratio of a ticker searching by company name.
    Args:
        query (str): company_name
    Returns:
        Price-to-earnings ratio (float)


finance_get_info_by_ticker(query)
    Return meta data of a ticker.
    Args:
        query (str): ticker_name:
    Returns:
        General information regarding the company.


finance_get_info_by_name(query)
    Return meta data of a ticker searching by company name.
    Args:
        query (str): company_name:
    Returns:
        General information regarding the company.


music_search_artist_entity_by_name(query)
    Return the fuzzy matching results of the query (artist name).
    Args:
        query (str): artist name
    Returns:
        Top-10 similar entity name in a list


music_search_song_entity_by_name(query)
    Return the fuzzy matching results of the query (song name).
    Args:
        query (str): song name
    Returns:
        Top-10 similar entity name in a list


music_get_billboard_rank_date(rank, date='')
    Return the song name(s) and the artist name(s) of a certain rank on a certain date; If no date is given, return the list of of a certain rank of all dates.
    Args:
        rank (int): the interested rank in billboard; from 1 to 100.
        date (Optional, str, in YYYY-MM-DD format): the interested date; leave it blank if do not want to specify the date.
    Returns:
        rank_list (list): a list of song names of a certain rank (on a certain date).
        artist_list (list): a list of author names corresponding to the song names returned.


music_get_billboard_attributes(date, attribute, song_name)
    Return the attributes of a certain song on a certain date
    Args:
        date (str, in YYYY-MM-DD format): the interested date of the song
        attribute (str): attributes from ['rank_last_week', 'weeks_in_chart', 'top_position', 'rank']
        song_name (str): the interested song name
    Returns:
        the value (str) of the interested attribute of a song on a certain date


music_grammy_get_best_artist_by_year(query)
    Return the Best New Artist of a certain year in between 1958 and 2019
    Args:
        query (int, in YYYY format): the interested year
    Returns:
        the list of artists who win the award


music_grammy_get_award_count_by_artist(query)
    Return the number of awards won by a certain artist between 1958 and 2019
    Args:
        query (str): the name of the artist
    Returns:
        the number of total awards (int)


music_grammy_get_award_count_by_song(query)
    Return the number of awards won by a song between 1958 and 2019
    Args:
        query (str): the name of the song
    Returns:
        the number of total awards (int)


music_grammy_get_best_song_by_year(query)
    Return the Song Of The Year in a certain year between 1958 and 2019
    Args:
        query (int, in YYYY format): the interested year
    Returns:
        the list of the song names that win the Song Of The Year in a certain year


music_grammy_get_award_date_by_artist(query)
    Return the award winning years of a certain artist
    Args:
        query (str): the name of the artist
    Returns:
        the list of years the artist is awarded


music_grammy_get_best_album_by_year(query)
    Return the Album Of The Year of a certain year between 1958 and 2019
    Args:
        query (int, in YYYY format): the interested year
    Returns:
        the list of albums that won the Album Of The Year in a certain year


music_grammy_get_all_awarded_artists()
    Return all the artists ever awarded Grammy Best New Artist between 1958 and 2019
    Returns:
        the list of artist ever awarded Grammy Best New Artist (list)


music_get_artist_birth_place(query)
    Return the birth place country code (2-digit) for the input artist
    Args:
        query (str): the name of the artist
    Returns:
        the two-digit country code following ISO-3166 (str)


music_get_artist_birth_date(query)
    Return the birth date of the artist
    Args:
        query (str): the name of the artist
    Returns:
        life_span_begin (str, in YYYY-MM-DD format if possible): the birth date of the person or the begin date of a band


music_get_members(query)
    Return the member list of a band
    Args:
        query (str): the name of the band
    Returns:
        the list of members' names.


music_get_lifespan(query)
    Return the lifespan of the artist
    Args:
        query (str): the name of the artist
    Returns:
        the birth and death dates in a list


music_get_song_author(query)
    Return the author of the song
    Args:
        query (str): the name of the song
    Returns:
        the author of the song (str)


music_get_song_release_country(query)
    Return the release country of the song
    Args:
        query (str): the name of the song
    Returns:
        the two-digit country code following ISO-3166 (str)


music_get_song_release_date(query)
    Return the release date of the song
    Args:
        query (str): the name of the song
    Returns:
        the date of the song (str in YYYY-MM-DD format)


music_get_artist_all_works(query)
    Return the list of all works of a certain artist
    Args:
        query (str): the name of the artist
    Returns:
        the list of all work names


sports_soccer_get_games_on_date(date, team_name=None)
    Get soccer games given date  
    Args:
        date (str, in YYYY-MM-DD/YYYY-MM/YYYY format): e.g., 2024-03-01, 2024-03, 2024
        team_name (Optional, str)
    Returns:
        info of the games, such as
            venue: whether the team is home or away in game
            result: win lose result of the game
            GF: goals of the team in game
            opponent: opponent of the team
            Captain: Captain of the team


sports_nba_get_games_on_date(date, team_name=None)
    Get all nba game rows given date_str
    Args:
        date (str, in YYYY-MM-DD/YYYY-MM/YYYY format): the time of the games, e.g. 2023-01-01, 2023-01, 2023
        team_name (Optional, str): basketball team name, like Los Angeles Lakers
    Returns:
        info of the games found, such as
            game_id: id of the game
            team_name_home: home team name
            team_name_away: away team name
            wl_home: win lose result of home team
            wl_away: win lose result of away team
            pts_home: home team points in the game
            pts_away: away team points in the game


sports_nba_get_play_by_play_data_by_game_ids(query)
    Get all nba play by play rows given game ids
    Args:
        game_ids (List[str]): nba game ids, e.g., ["0022200547", "0029600027"]
    Returns:
        info of the play by play events of given game id
</DOCUMENTATION>
""".strip()

EVAL_SYSTEM_INSTRUCTIONS = """
You are a helpful, respectful and honest question answering system. I will be providing you with questions, as well as some additional context that may be helpful in answering those questions. You will be given the original question "ORIGINAL QUESTION" and answer the "QUESTION", which will ask "Is <answer 1> equivalent to <answer 2>?". "Equivalent" here means that the answers are both suitable in terms of the

If the answers are equivalent, you will simply output as "ANSWER: 1". If they are not equivalent, you will simply output "ANSWER: 0"
""".strip()


CONTEXTUAL_EXAMPLES = """
# Example 1
QUESTION: who was the first nba player to get a recorded triple double in basketball?
<DOCS>
<DOC>
Andy Phillip got a triple-double versus the Fort Wayne Pistons on December 14, 1950.
</DOC>
</DOCS>
ANSWER: andy phillip

# Example 2
QUESTION: what year did ronald reagan become kommissar in the soviet union?
<DOCS>
<DOC>
Ronald Wilson Reagan was an American politician and actor who served as the 40th president of the United States from 1981 to 1989.
</DOC>
<DOC>
Reagan's policies also helped contribute to the end of the Cold War and the end of Soviet communism.
</DOC>
</DOCS>
ANSWER: invalid question

# Example 3
QUESTION: what are the total number of farms in nebraska?
<DOCS>
<DOC>
There are many farms in Nebraska, some growing corn, while other growing sugar beats.
</DOC>
<DOC>
Nebraska's largest industries are the argiculture and cattle production industries.
</DOC>
</DOCS>
ANSWER: i don't know
""".strip()

NO_CONTEXT_CONTEXTUAL_EXAMPLES = """
# Example 1
QUESTION: who was the voice of woody in toy story?
ANSWER: Tom Hanks

# Example 2
QUESTION: what year did ronald reagan become kommissar in the soviet union?
ANSWER: Invalid Question

# Example 3
QUESTION: is brad pitt older than selena gomez?
ANSWER: Yes

# Example 4
QUESTION: what is the age difference between angelina jolie and billy bob thornton?
ANSWER: 20 years.

# Example 5
QUESTION: what is the area of a square that is 10 inches wide and 12 inches tall?
ANSWER: 120 inches

# Example 6
QUESTION: what was the closing price of nvidia yesterday?
ANSWER: I don't know

# Example 7
QUESTION: what is the distance of a marathon race?
ANSWER: 26 miles

# Example 8
QUESTION: if I have $67,312 and I earn 7\% interest per year compounded yearly, how much will I have after 3 years
ANSWER: I don't know

# Example 9
QUESTION: how many grammy wins does beyoncé have in total?
ANSWER: I don't know

# Example 10
QUESTION: what date was internet explorer released?
ANSWER: I don't know

# Example 11
QUESTION: how far does a bullet travel before it falls onto the ground?
ANSWER: I don't know

Remember, only answer something other than "I don't know" if you are confident you know the correct answer.
""".strip()



def prompt_format_v0(prompt, query, query_time, candidates, include_context_examples, tokenizer, token_budget=3500):
    bos = "<|begin_of_text|>"
    sys_message = "<|start_header_id|>system<|end_header_id|>"
    eos = "<|eot_id|>"
    user_message = "<|start_header_id|>user<|end_header_id|>"
    assistant_message = "<|start_header_id|>assistant<|end_header_id|>"
    segments_text = [f"<DOC>\n{segment[0]}\n</DOC>" for segment in candidates]
    segments_text = '\n'.join(segments_text).strip()
    text_representation = f"QUESTION (asked at {query_time}): {query}\n<DOCS>\n{segments_text}\n</DOCS>"
    text_representation = tokenizer.decode(tokenizer.encode(text_representation, add_special_tokens=False)[:token_budget])
    if include_context_examples:
        formatted_text = f"{bos}{sys_message}\n{prompt}{eos}{user_message}\n{CONTEXTUAL_EXAMPLES}\n{text_representation}{eos}{assistant_message}ANSWER:"
    else:
        formatted_text = f"{bos}{sys_message}\n{prompt}{eos}{user_message}\n{text_representation}{eos}{assistant_message}ANSWER:"
    return formatted_text


def prompt_format_v1(prompt, query, query_time, candidates, include_context_examples, tokenizer, token_budget=3500):
    bos = "<|begin_of_text|>"
    sys_message = "<|start_header_id|>system<|end_header_id|>"
    eos = "<|eot_id|>"
    user_message = "<|start_header_id|>user<|end_header_id|>"
    assistant_message = "<|start_header_id|>assistant<|end_header_id|>"
    segments_text = [f"<DOC>\n{segment[0]}\n</DOC>" for segment in reversed(candidates)]
    segments_text = '\n'.join(segments_text).strip()
    text_representation = f"\n<DOCS>\n{segments_text}\n</DOCS>\nQUESTION (asked at {query_time}): {query}"
    text_representation = tokenizer.decode(tokenizer.encode(text_representation, add_special_tokens=False)[-token_budget:])
    if include_context_examples:
        formatted_text = f"{bos}{sys_message}\n{prompt}{eos}{user_message}\n{CONTEXTUAL_EXAMPLES}\n{text_representation}{eos}{assistant_message}ANSWERABLE:"
    else:
        formatted_text = f"{bos}{sys_message}\n{prompt}{eos}{user_message}\n{text_representation}{eos}{assistant_message}ANSWERABLE:"
    return formatted_text


def prompt_format_no_context(query, query_time, include_examples, tokenizer, token_budget=3500):
    bos = "<|begin_of_text|>"
    sys_message = "<|start_header_id|>system<|end_header_id|>"
    eos = "<|eot_id|>"
    user_message = "<|start_header_id|>user<|end_header_id|>"
    assistant_message = "<|start_header_id|>assistant<|end_header_id|>"
    text_representation = f"\nQUESTION (asked at {query_time}): {query}"
    text_representation = tokenizer.decode(tokenizer.encode(text_representation, add_special_tokens=False)[-token_budget:])
    if include_examples:
        formatted_text = f"{bos}{sys_message}\n{QA_SYSTEM_INSTRUCTIONS_NO_CONTEXT}{eos}{user_message}\n{NO_CONTEXT_CONTEXTUAL_EXAMPLES}\n{text_representation}{eos}{assistant_message}ANSWER:"
    else:
        formatted_text = f"{bos}{sys_message}\n{QA_SYSTEM_INSTRUCTIONS_NO_CONTEXT}{eos}{user_message}\n{text_representation}{eos}{assistant_message}ANSWER:"
    return formatted_text


class LlamaLLM:
    def __init__(self, model_name=MODELS_DIR / "Meta-Llama-3-8B-Instruct", peft_path=MODELS_DIR / "peft_task3", api_peft_path=MODELS_DIR / "peft_api", use_peft=True, qa_prompt=QA_SYSTEM_INSTRUCTIONS, api_prompt=API_SYSTEM_INSTRUCTIONS ,bnb_config=DEFAULT_CONFIG, batch_size=1, prompt_format="v1", include_context_examples=True):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
        )
        self.use_peft = use_peft
        self.peft_model = None
        if use_peft:
            self.model.load_adapter(peft_path, adapter_name="context")
            self.model.load_adapter(api_peft_path, adapter_name="api")
        self.qa_prompt = qa_prompt
        self.api_prompt = api_prompt
        self.batch_size = batch_size
        self.model.generation_config.pad_token_ids = self.tokenizer.pad_token_id
        self.generation_pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=100,
            batch_size=batch_size,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            truncation=True
        )
        self.prompt_format = prompt_format
        self.include_context_examples = include_context_examples

    """
    Reference prompt structure
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    """

    # def process_example(self, text):
    #     with torch.no_grad():
    #         bos = "<|begin_of_text|>"
    #         sys_message = "<|start_header_id|>system<|end_header_id|>"
    #         eos = "<|eot_id|>"
    #         user_message = "<|start_header_id|>user<|end_header_id|>"
    #         assistant_message = "<|start_header_id|>assistant<|end_header_id|>"
    #         formatted_text = f"{bos}{sys_message}\n{self.qa_prompt}{eos}{user_message}\n{text[:self.max_len]}{eos}{assistant_message}ANSWER:"
    #         result = self.generation_pipe(formatted_text)
    #         result = result[0]["generated_text"].split("ANSWER:")[-1].strip()
    #         return result

    # def process_no_candidates(self, query, query_time):
    #     formatted_text = prompt_format_no_context(query, query_time, self.include_context_examples, self.tokenizer)
    #     result = self.generation_pipe(formatted_text)
    #     result = result[0]["generated_text"].split("ANSWER:")[-1].strip()
    #     return result

    def process_candiates(self, query, query_time, candidates):
        self.model.enable_adapters()
        self.model.set_adapter("context")
        with torch.no_grad():
            if self.prompt_format == "v0":
                formatted_text = prompt_format_v0(self.qa_prompt, query, query_time, candidates, self.include_context_examples, self.tokenizer)
            elif self.prompt_format == "v1":
                formatted_text = prompt_format_v1(self.qa_prompt, query, query_time, candidates, self.include_context_examples, self.tokenizer)
            else:
                formatted_text = prompt_format_no_context(query, query_time, self.include_context_examples, self.tokenizer)
            # formatted_text = formatted_text.lower()
            result = self.generation_pipe(formatted_text)
            # result = result[0]["generated_text"].split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
            result = result[0]["generated_text"].split("ANSWER:")[-1].strip()
            if not result:
                result = "i don't know"
                # result = "NO RESULT"

            # if result == "i don't know":
            #     return self.process_no_candidates(query, query_time)
            return result
        ##################
        
    # def process_candiates(self, query, query_time, candidates):
    #     self.model.enable_adapters()
    #     self.model.set_adapter("context")
    #     with torch.no_grad():
    #         if self.prompt_format == "v0":
    #             formatted_text = prompt_format_v0(self.qa_prompt, query, query_time, candidates, self.include_context_examples, self.tokenizer)
    #         elif self.prompt_format == "v1":
    #             formatted_text = prompt_format_v1(self.qa_prompt, query, query_time, candidates, self.include_context_examples, self.tokenizer)
    #         else:
    #             formatted_text = prompt_format_no_context(query, query_time, self.include_context_examples, self.tokenizer)
    #         # formatted_text = formatted_text.lower()
    #         # Create a completion
    #         message_text = [{
    #             "role": "user",
    #             "content": formatted_text
    #         }]

    #         completion = client.chat.completions.create(
    #             model="gpt-35-turbo", # gpt-35-turbo
    #             messages=message_text,
    #             temperature=0.7,
    #             max_tokens=800,
    #             top_p=0.95,
    #             frequency_penalty=0,
    #             presence_penalty=0,
    #             stop=None
    #         )

    #         # print('user:', message_text[0]['content'])
    #         # print('assistant:', completion.choices[0].message.content)
    #         result = completion.choices[0].message.content
    #         # result = result[0]["generated_text"].split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    #         # result = result[0]["generated_text"].split("ANSWER:")[-1].strip()
    #         if not result:
    #             result = "i don't know"
    #             result = "NO RESULT"

    #         # if result == "i don't know":
    #         #     return self.process_no_candidates(query, query_time)
    #         return result
        ##########################

    def process_api(self, text):
        self.model.enable_adapters()
        self.model.set_adapter("api")
        with torch.no_grad():
            bos = "<|begin_of_text|>"
            sys_message = "<|start_header_id|>system<|end_header_id|>"
            eos = "<|eot_id|>"
            user_message = "<|start_header_id|>user<|end_header_id|>"
            assistant_message = "<|start_header_id|>assistant<|end_header_id|>"
            formatted_text = f"{bos}{sys_message}\n{self.api_prompt}{eos}{user_message}\n{API_DOCS}\n{text}{eos}{assistant_message}"
            result = self.generation_pipe(formatted_text)
            result = result[0]["generated_text"].split("<|start_header_id|>assistant<|end_header_id|>")[1].strip()
            return result


# if __name__ == "__main__":
#     llm = LlamaLLM(batch_size=4)
