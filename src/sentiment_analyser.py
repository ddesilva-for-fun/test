import os
from src.custom_types import LlmConfig
import openai
from openai.types.responses import Response
import logging

LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL", "INFO")
logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)

class SentimentAnalyser():
    
    def __init__(self, *,  input: str, llm_client: openai.OpenAI, config: LlmConfig):
        self.input = input
        self._llm_client = llm_client
        self._config = config

    def analyse(self, messages):
        try:
            print(messages)
            llm_response: Response = self._llm_client.responses.create(
                model=self._config.llm,
                input=messages,
                store=False,
                stream=False,
                timeout=60,
                top_p=self._config.top_p,
                temperature=self._config.temperature,
                max_output_tokens=self._config.max_tokens,
            )
            logger.debug(f"LLM Response: {llm_response}")
            message = [i for i in llm_response.model_dump().get("output") if i.get("type") == "message"]
            output = [i.get("text") for i in message[0].get("content") if i.get("type") == "output_text"][0]
            return output
        except openai.OpenAIError as e:
            logger.error(f"ERROR: {e}")
            raise
        except Exception as e:
            logger.error(f"ERROR: An unexpected error occured: {e}")
            raise

    @staticmethod
    def render_prompt():
        return """
        ## Task

        You are a sentiment analyser, tasked with analysing comments from different website posts. 

        ## Input

        A comment from a website post whose sentiment you are to analyse.

        ## Output

        You should simply output a sentiment score which is a float between -1 and 1 as well as a label which will be a string with values of "NEGATIVE", "POSITIVE" or "NEUTRAL". 

        -1 signifies an extemely negative sentiment in the comment, 0 represents a neutral sentiment and 1 signifies an extremely positive sentiment in the comment.

        If the sentiment score is less than 0, the label should be equal to "NEGATIVE".
        If the sentiment score is equal to 0, the label should be equal to "NEUTRAL".
        If the sentiment score is more than 0, the label should be equal to "POSITIVE".

        The output format should be a JSON object with keys of "sentimentScore" and "label". An example output would be: {"sentimentScore": 0, "label": "NEUTRAL"}

        """

    def prepare_llm_input(self):
        try:
            messages = [{"role": "system", "content": SentimentAnalyser.render_prompt()}]
            messages.append({"role": "user", "content": self.input, "type": "message"})
            logger.debug(f"Prepared LLM messages: {messages}")
            return messages
        except Exception as e:
            logger.error(f"Failed to prepare messages: Error: {e}")


    
            