Simple implementation of a sentiment analyser class using OpenAI's Responses API - docs found here - https://platform.openai.com/docs/api-reference/responses

# Example Usage:
sentiment_analyser = SentimentAnalyser(input="Comment from website post", llm_client=openai_cli, config=config) # where openai_cli is of type OpenAI and config is of type LlmConfig found in src/custom_types

response_dict = sentiment_analyser.analyse(messages=sentiment_analyser.prepare_llm_input())

sentiment_score = response_dict.get("sentimentScore") # example: 0

label = response_dict.get("sentimentScore") # example: "NEUTRAL"

# Tests:
Only tests for the happy and unhappy paths of the analyse method of the SentimentAnalyser class for the time being.

To run the tests, create a venv, pip install the requirements in the src folder and run `pytest` from the root directory.

# Config:
See OpenAI docs for more info on the variables (top_p, temperature, max_tokens, llm) in the LlmConfig which can be changed as required via the config instance variable on the SentimentAnalyser class. 


