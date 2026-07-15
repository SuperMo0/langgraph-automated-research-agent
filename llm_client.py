from typing import TypeVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam, ParsedChatCompletion
from openai.types.chat.chat_completion_function_tool_param import ChatCompletionFunctionToolParam
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import OPENAI_API_KEY, OPENAI_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

_ResponseFormatT = TypeVar("_ResponseFormatT", bound=BaseModel)

_RETRYABLE_OPENAI_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
_retry_openai = retry(
    retry=retry_if_exception_type(_RETRYABLE_OPENAI_ERRORS),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)


@_retry_openai
def create_chat_completion(
    messages: list[ChatCompletionMessageParam],
    tools: list[ChatCompletionFunctionToolParam] | None = None,
) -> ChatCompletion:
    if tools is None:
        return client.chat.completions.create(model=OPENAI_MODEL, messages=messages)
    return client.chat.completions.create(
        model=OPENAI_MODEL, messages=messages, tools=tools, parallel_tool_calls=True
    )


@_retry_openai
def parse_chat_completion(
    messages: list[ChatCompletionMessageParam],
    response_format: type[_ResponseFormatT],
) -> ParsedChatCompletion[_ResponseFormatT]:
    return client.chat.completions.parse(model=OPENAI_MODEL, messages=messages, response_format=response_format)
