from openai import OpenAI
from os import environ
from text_loader import *


openai_api_key = environ['OPENAI_API_KEY']


def generate_response_with_api(model_name, message):
    """
    Generate a response to a given message using the model and tokenizer.
    message: a dict with 'role' and 'content' keys so that chat templates can be applied.
    """
    client = OpenAI(api_key=openai_api_key)

    completion = client.chat.completions.create(
        model=model_name,
        messages=message
    )

    return completion.choices[0].message.content



def post_process_win_response(response):
    response = response.strip().lower()
    if 'text1' not in response and 'text2' not in response and 'tie' not in response:
        winner = 'tie'
    elif 'text1' in response and 'text2' in response:
        winner = 'tie'
    elif 'text1' in response:
        winner = 'text1'
    elif 'text2' in response:
        winner = 'text2'
    else:
        winner = 'tie'
    return winner

        

def template_paraphrase_quality(model_name, original_text, paraphrased_text):
    system_prompt = """
    You are an expert linguist and paraphrase evaluator. Your task is to assess the quality of a paraphrased text compared to the original source text. Use the following scoring criteria:

    5 - Approximately equivalent: Meaning is preserved; differences are only in wording or structure.
    4 - Nearly equivalent: Meaning is mostly preserved; minor factual details differ.
    3 - Somewhat equivalent: Some meaning is preserved; important details or meanings differ.
    2 - Topically related: The texts are on the same topic but most meaning is lost.
    1 - Not topically related: The texts are not related in topic or meaning.

    Provide your final output as a JSON object in this format:
    {
    "score": <score from 1 to 5>,
    "justification": "<brief explanation>"
    }
    """

    user_prompt = f"""
    Evaluate the following paraphrase using the criteria above:

    Original Text:
    \"\"\"{original_text}\"\"\"

    Paraphrased Text:
    \"\"\"{paraphrased_text}\"\"\"

    What score (1 to 5) would you assign to this paraphrase, and why?
    """


    if 'gpt' in model_name:
        return [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    else:
        #TO DO
        raise ValueError("Unsupported model name. Please use a valid model name.")

def template_win_rate(model_name, text1, text2):
    system_prompt = """
    You are a neutral and highly capable evaluator. Your task is to compare two versions of a text and select the one that communicates the ideas more clearly, accurately, and fluently. Consider:

    - Fidelity of meaning (how clearly the core message is conveyed)
    - Clarity and conciseness
    - Grammatical correctness and fluency
    - Naturalness and appropriateness of phrasing

    
    Give your vote solely based on quality. If one of them is a CLEAR winner, ONLY then vote for that one. Otherwise, vote for `tie`.

    Respond with **only one of the following**, and nothing else:
    - text1
    - text2
    - tie
    """

    user_prompt = f"""
    Compare the following two texts and give your vote depending on meaning clarity, fluency, and overall quality. If one of them is a CLEAR winner, ONLY then vote for that one. Otherwise, vote for `tie`. Respond with one of these 3 options: `text1`, `text2`, `tie`.

    Text 1:
    \"\"\"{text1}\"\"\"

    Text 2:
    \"\"\"{text2}\"\"\"
    """


    if 'gpt' in model_name:
        return [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    else:
        #TO DO
        raise ValueError("Unsupported model name. Please use a valid model name.")


