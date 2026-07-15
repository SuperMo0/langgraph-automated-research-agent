from pydantic import BaseModel
from requests.exceptions import ConnectionError, HTTPError
from tavily import TavilyClient
from tavily.errors import TimeoutError as TavilyTimeoutError
from tavily.errors import UsageLimitExceededError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import TAVILY_API_KEY


class SearchResult(BaseModel):
    title: str
    url: str
    content: str


class SearchResponse(BaseModel):
    query: str
    answer: str | None = None
    results: list[SearchResult]


class SearchError(Exception):
    """Raised when Tavily search fails, including after retries are exhausted."""


_client = TavilyClient(api_key=TAVILY_API_KEY)

_RETRYABLE_ERRORS = (TavilyTimeoutError, ConnectionError, HTTPError, UsageLimitExceededError)


@retry(
    retry=retry_if_exception_type(_RETRYABLE_ERRORS),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def _search(query: str, max_results: int) -> dict:
    return _client.search(query, max_results=max_results, include_answer=True)


def search_web(query: str, max_results: int = 5) -> SearchResponse:
    try:
        raw = _search(query, max_results)
    except Exception as exc:
        raise SearchError(str(exc)) from exc
    return SearchResponse.model_validate(raw)
