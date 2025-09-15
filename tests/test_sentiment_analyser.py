import json
from unittest.mock import patch
from src.custom_types import LlmConfig
from src.sentiment_analyser import SentimentAnalyser 
from openai import OpenAI, OpenAIError
import pytest
from openai.types.responses import Response
import os


def mock_openai_responses_api_response():
    # Example response from OpenAI Responses API docs - https://platform.openai.com/docs/api-reference/responses/create
    response_dict = {
        "id": "resp_67ccd3a9da748190baa7f1570fe91ac604becb25c45c1d41",
        "object": "response",
        "created_at": 1741476777,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": "gpt-4o-2024-08-06",
        "output": [
            {
            "type": "message",
            "id": "msg_67ccd3acc8d48190a77525dc6de64b4104becb25c45c1d41",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                "type": "output_text",
                "text": '{"sentimentScore": 0, "label": "NEUTRAL"}',
                "annotations": []
                }
            ]
            }
        ],
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {
            "effort": None,
            "summary": None
        },
        "store": True,
        "temperature": 1,
        "text": {
            "format": {
            "type": "text"
            }
        },
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 328,
            "input_tokens_details": {
            "cached_tokens": 0
            },
            "output_tokens": 52,
            "output_tokens_details": {
            "reasoning_tokens": 0
            },
            "total_tokens": 380
        },
        "user": None,
        "metadata": {}
    }
    return Response(**response_dict)

@patch("src.sentiment_analyser.openai.OpenAI.responses")
def test_example_usage(mock_openai_responses_response):
    # Arrange
    os.environ["OPENAI_API_KEY"] = "blah"
    os.environ["OPENAI_ORG_ID"] = "blah"
    os.environ["OPENAI_PROJECT_ID"] = "blah"
    os.environ["OPENAI_WEBHOOK_SECRET"] = "blah"
    openai_cli = OpenAI()
    config = LlmConfig(top_p=0.5, temperature=0.5, max_tokens=600, llm="gpt-4o-mini")
    sentiment_analyser = SentimentAnalyser(input="blah", llm_client = openai_cli, config=config)
    
    mock_openai_responses_response.create.return_value = mock_openai_responses_api_response()
    # Act
    response_dict = json.loads(sentiment_analyser.analyse(messages = sentiment_analyser.prepare_llm_input()))
    # Assert
    sentiment_score = response_dict.get("sentimentScore") # example: 0
    label = response_dict.get("label") # example: "NEUTRAL"
    assert sentiment_score == 0
    assert label == "NEUTRAL"

@patch("src.sentiment_analyser.openai.OpenAI.responses")
def test_analyse_method_happy_path(mock_openai_responses_response):
    # Arrange
    os.environ["OPENAI_API_KEY"] = "blah"
    os.environ["OPENAI_ORG_ID"] = "blah"
    os.environ["OPENAI_PROJECT_ID"] = "blah"
    os.environ["OPENAI_WEBHOOK_SECRET"] = "blah"
    openai_cli = OpenAI()
    config = LlmConfig(top_p=0.5, temperature=0.5, max_tokens=600, llm="gpt-4o-mini")
    sentiment_analyser = SentimentAnalyser(input="blah", llm_client = openai_cli, config=config)

    mock_openai_responses_response.create.return_value = mock_openai_responses_api_response()
    # Act
    analyse_response = sentiment_analyser.analyse(messages="blah")
    # Assert
    assert analyse_response == '{"sentimentScore": 0, "label": "NEUTRAL"}'
    

@patch("src.sentiment_analyser.openai.OpenAI.responses")
def test_analyse_method_OpenAIError(mock_openai_responses_response):
    # Arrange
    os.environ["OPENAI_API_KEY"] = "blah"
    os.environ["OPENAI_ORG_ID"] = "blah"
    os.environ["OPENAI_PROJECT_ID"] = "blah"
    os.environ["OPENAI_WEBHOOK_SECRET"] = "blah"
    openai_cli = OpenAI()
    config = LlmConfig(top_p=0.5, temperature=0.5, max_tokens=600, llm="gpt-4o-mini")
    sentiment_analyser = SentimentAnalyser(input="blah", llm_client = openai_cli, config=config)
    mock_openai_responses_response.create.side_effect = OpenAIError()
    # Act and Assert
    with pytest.raises(OpenAIError):
        analyse_response = sentiment_analyser.analyse(messages="blah")
    
@patch("src.sentiment_analyser.openai.OpenAI.responses")
def test_analyse_method_unknown_exception(mock_openai_responses_response):
    # Arrange
    os.environ["OPENAI_API_KEY"] = "blah"
    os.environ["OPENAI_ORG_ID"] = "blah"
    os.environ["OPENAI_PROJECT_ID"] = "blah"
    os.environ["OPENAI_WEBHOOK_SECRET"] = "blah"
    openai_cli = OpenAI()
    config = LlmConfig(top_p=0.5, temperature=0.5, max_tokens=600, llm="gpt-4o-mini")
    sentiment_analyser = SentimentAnalyser(input="blah", llm_client = openai_cli, config=config)
    mock_openai_responses_response.create.side_effect = Exception()
    # Act and Assert
    with pytest.raises(Exception):
        analyse_response = sentiment_analyser.analyse(messages="blah")
    
