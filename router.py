from openai.types.chat import ChatCompletionMessageParam

from llm_client import parse_chat_completion
from models import RouterDecision

_SYSTEM_PROMPT = """Classify the user's research query into exactly one type:

- factual: a question with a definite correct answer (e.g. "When did X happen?", "What is \
the population of Y?"). Prioritise finding primary sources.
- analytical: a comparison or evaluation between things (e.g. "How does X compare to Y?", \
"What are the tradeoffs between X and Y?"). Prioritise gathering multiple perspectives and \
data points.
- opinion: a "should I" or "what do you think" query seeking judgment or a recommendation \
(e.g. "Should I use X or Y?", "Is X worth it?"). Prioritise gathering expert opinions and a \
balanced view.

Also give your confidence in this classification (high/medium/low) and a one-sentence \
reasoning."""


def classify_query(query: str) -> RouterDecision:
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    response = parse_chat_completion(messages, RouterDecision)
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("Router model did not return a structured decision.")
    return parsed


if __name__ == "__main__":
    test_queries = [
        ("When was the Eiffel Tower built?", "factual"),
        ("How does PydanticAI compare to LangGraph?", "analytical"),
        ("Should I learn Rust or Go in 2026?", "opinion"),
    ]
    for query, expected in test_queries:
        decision = classify_query(query)
        status = "OK" if decision.query_type == expected else "MISMATCH"
        print(f"[{status}] {query!r} -> {decision.query_type} (confidence={decision.confidence})")
        print(f"  expected={expected} reasoning={decision.reasoning}")
