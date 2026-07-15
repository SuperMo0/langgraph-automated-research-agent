import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
    pydantic_function_tool,
)
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
)
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_function_tool_param import (
    ChatCompletionFunctionToolParam,
)
from openai.types.chat.chat_completion_message_function_tool_call_param import (
    ChatCompletionMessageFunctionToolCallParam,
)
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import MAX_RESEARCH_ITERATIONS, OPENAI_API_KEY, OPENAI_MODEL
from models import EvidenceStore, QueryType
from search_client import SearchError, search_web

_BASE_INSTRUCTIONS = """You are a research agent. Investigate the user's question using the \
available tools:

1. search_web to find sources for each sub-question
2. summarise_content to condense long search results, focused on what's relevant
3. check_fact to verify any claim that seems uncertain before storing it
4. store_evidence to save each confirmed piece of evidence under a short, descriptive key

Break the question into sub-questions first, then work through them. When you have \
gathered sufficient evidence, respond with a plain text message (no tool call) starting \
with "RESEARCH COMPLETE" followed by a brief summary of what you found."""

_SYSTEM_PROMPTS: dict[QueryType, str] = {
    "factual": _BASE_INSTRUCTIONS
    + "\n\nThis is a FACTUAL question with a definite correct answer. Prioritise finding "
    "and citing primary sources.",
    "analytical": _BASE_INSTRUCTIONS
    + "\n\nThis is an ANALYTICAL question comparing or evaluating things. Prioritise "
    "gathering multiple perspectives and concrete data points.",
    "opinion": _BASE_INSTRUCTIONS
    + "\n\nThis is an OPINION-seeking question. Gather expert opinions, cite sources for "
    "each, and present a balanced view rather than a single verdict.",
}

class SearchWebArgs(BaseModel):
    """Search the web for information relevant to a query."""

    query: str


class SummariseContentArgs(BaseModel):
    """Summarise a piece of content, focused on a specific aspect."""

    content: str
    focus: str


class CheckFactArgs(BaseModel):
    """Check whether a claim is supported by the given context."""

    claim: str
    context: str


class GetCurrentDateArgs(BaseModel):
    """Get today's date in ISO format."""


class StoreEvidenceArgs(BaseModel):
    """Store a confirmed piece of evidence under a short, descriptive key."""

    key: str
    content: str


TOOLS: list[ChatCompletionFunctionToolParam] = [
    pydantic_function_tool(SearchWebArgs, name="search_web"),
    pydantic_function_tool(SummariseContentArgs, name="summarise_content"),
    pydantic_function_tool(CheckFactArgs, name="check_fact"),
    pydantic_function_tool(GetCurrentDateArgs, name="get_current_date"),
    pydantic_function_tool(StoreEvidenceArgs, name="store_evidence"),
]

_RETRYABLE_OPENAI_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)


@retry(
    retry=retry_if_exception_type(_RETRYABLE_OPENAI_ERRORS),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def _create_chat_completion(
    client: OpenAI,
    messages: list[ChatCompletionMessageParam],
    tools: list[ChatCompletionFunctionToolParam] | None = None,
) -> ChatCompletion:
    if tools is None:
        return client.chat.completions.create(model=OPENAI_MODEL, messages=messages)
    return client.chat.completions.create(
        model=OPENAI_MODEL, messages=messages, tools=tools, parallel_tool_calls=True
    )


def _to_assistant_param(message: ChatCompletionMessage) -> ChatCompletionAssistantMessageParam:
    param: ChatCompletionAssistantMessageParam = {"role": "assistant", "content": message.content}
    function_calls = [call for call in message.tool_calls or [] if call.type == "function"]
    if function_calls:
        param["tool_calls"] = [
            ChatCompletionMessageFunctionToolCallParam(
                id=call.id,
                type="function",
                function={"name": call.function.name, "arguments": call.function.arguments},
            )
            for call in function_calls
        ]
    return param


def get_current_date() -> str:
    return date.today().isoformat()


def _search_web_tool(query: str) -> str:
    try:
        response = search_web(query)
    except SearchError as exc:
        return f"search unavailable: {exc}"
    if not response.results:
        return "No results found."
    return "\n".join(f"- {r.title} ({r.url}): {r.content}" for r in response.results)


def _summarise_content_tool(client: OpenAI, content: str, focus: str) -> str:
    response = _create_chat_completion(
        client,
        [
            {
                "role": "system",
                "content": "Summarise the given content in 2-4 sentences, focused specifically "
                "on the requested aspect. Be concise and factual.",
            },
            {"role": "user", "content": f"Focus: {focus}\n\nContent:\n{content}"},
        ],
    )
    return response.choices[0].message.content or ""


def _check_fact_tool(client: OpenAI, claim: str, context: str) -> str:
    response = _create_chat_completion(
        client,
        [
            {
                "role": "system",
                "content": "You are a fact-checker. Given a claim and supporting context, judge "
                "whether the context supports the claim. Respond with SUPPORTED, UNSUPPORTED, or "
                "UNCERTAIN followed by a one-sentence explanation.",
            },
            {"role": "user", "content": f"Claim: {claim}\n\nContext:\n{context}"},
        ],
    )
    return response.choices[0].message.content or ""


def _store_evidence_tool(evidence: EvidenceStore, key: str, content: str) -> str:
    evidence[key] = content
    return f"Stored evidence under key '{key}' ({len(content)} chars)."


def _dispatch_tool(
    client: OpenAI,
    call: ChatCompletionMessageFunctionToolCall,
    evidence: EvidenceStore,
) -> str:
    raw_args = call.function.arguments
    try:
        match call.function.name:
            case "search_web":
                return _search_web_tool(SearchWebArgs.model_validate_json(raw_args).query)
            case "summarise_content":
                args = SummariseContentArgs.model_validate_json(raw_args)
                return _summarise_content_tool(client, args.content, args.focus)
            case "check_fact":
                args = CheckFactArgs.model_validate_json(raw_args)
                return _check_fact_tool(client, args.claim, args.context)
            case "get_current_date":
                GetCurrentDateArgs.model_validate_json(raw_args)
                return get_current_date()
            case "store_evidence":
                args = StoreEvidenceArgs.model_validate_json(raw_args)
                return _store_evidence_tool(evidence, args.key, args.content)
            case _:
                return f"error: unknown tool '{call.function.name}'"
    except ValidationError as exc:
        return f"error: invalid arguments for '{call.function.name}': {exc}"


def _dispatch_tool_calls(
    client: OpenAI,
    calls: list[ChatCompletionMessageFunctionToolCall],
    evidence: EvidenceStore,
) -> list[str]:
    with ThreadPoolExecutor(max_workers=len(calls)) as executor:
        futures = [executor.submit(_dispatch_tool, client, call, evidence) for call in calls]
        return [future.result() for future in futures]


def _truncate(text: str, limit: int = 60) -> str:
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def run_research_agent(query: str, query_type: QueryType, *, verbose: bool = False) -> EvidenceStore:
    client = OpenAI(api_key=OPENAI_API_KEY)
    evidence: EvidenceStore = {}
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": _SYSTEM_PROMPTS[query_type]},
        {"role": "user", "content": query},
    ]

    step = 0
    for _ in range(MAX_RESEARCH_ITERATIONS):
        response = _create_chat_completion(client, messages, TOOLS)
        message = response.choices[0].message
        messages.append(_to_assistant_param(message))

        function_calls = [call for call in message.tool_calls or [] if call.type == "function"]
        if not function_calls:
            step += 1
            if verbose:
                print(f"  Iter {step}: RESEARCH COMPLETE — gathered {len(evidence)} evidence items")
            break

        results = _dispatch_tool_calls(client, function_calls, evidence)
        for call, result in zip(function_calls, results):
            step += 1
            if verbose:
                args_preview = _truncate(call.function.arguments)
                print(f"  Iter {step}: {call.function.name}({args_preview}) → {len(result)} chars")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    else:
        if verbose:
            print(f"  Reached max iterations ({MAX_RESEARCH_ITERATIONS}) without an explicit completion signal.")

    return evidence


if __name__ == "__main__":
    test_query = sys.argv[1] if len(sys.argv) > 1 else "What are the main differences between PydanticAI and LangGraph?"
    result = run_research_agent(test_query, "analytical", verbose=True)
    print(f"\nGathered {len(result)} evidence items:")
    for key, content in result.items():
        print(f"  {key}: {content[:100]}")
