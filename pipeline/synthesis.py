from datetime import UTC, datetime

from openai.types.chat import ChatCompletionMessageParam

from core.evidence import format_evidence
from core.llm_client import parse_chat_completion
from core.models import EvidenceStore, QueryType, ResearchReport

_SYSTEM_PROMPT = """You are a research report writer. Given a research question, its type \
(factual / analytical / opinion), and evidence gathered by a research agent, produce a \
structured research report.

Rules:
- Ground every claim in the provided evidence; do not invent facts not supported by it.
- key_findings should be 5-10 concise bullet points.
- sources should list the distinct sources referenced in the evidence, extracting title \
and url from the evidence text where available; omit the url if it isn't given.
- confidence should reflect how well the evidence actually supports the findings: "high" \
if well-corroborated by multiple sources, "medium" if partially supported, "low" if thin \
or based on a single weak source.
- limitations should honestly state what the report does not cover, given the evidence \
available."""


def _build_messages(
    query: str,
    query_type: QueryType,
    evidence: EvidenceStore,
    previous_report: ResearchReport | None,
    feedback: str | None,
) -> list[ChatCompletionMessageParam]:
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Query: {query}\nQuery type: {query_type}\n\n"
            f"Evidence:\n{format_evidence(evidence)}",
        },
    ]
    if previous_report is not None and feedback is not None:
        messages.append(
            {
                "role": "user",
                "content": "Your previous draft scored below the quality threshold.\n\n"
                f"Previous draft:\n{previous_report.model_dump_json(indent=2)}\n\n"
                f"Evaluator feedback: {feedback}\n\n"
                "Revise the report to address this feedback, using the evidence above.",
            }
        )
    return messages


def synthesise_report(
    query: str,
    query_type: QueryType,
    evidence: EvidenceStore,
    *,
    previous_report: ResearchReport | None = None,
    feedback: str | None = None,
) -> ResearchReport:
    messages = _build_messages(query, query_type, evidence, previous_report, feedback)
    response = parse_chat_completion(messages, ResearchReport)
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("Synthesis model did not return a structured report.")
    return parsed.model_copy(
        update={
            "query": query,
            "query_type": query_type,
            "generated_at": datetime.now(UTC).isoformat(),
        }
    )


if __name__ == "__main__":
    mock_evidence: EvidenceStore = {
        "pydanticai_typing": "PydanticAI enforces typed result models via Pydantic, so "
        "agents cannot return unvalidated output. Source: PydanticAI docs "
        "(ai.pydantic.dev/agents/).",
        "langgraph_state": "LangGraph models workflows as explicit state graphs with visible "
        "nodes and edges, and has native human-in-the-loop support via interrupt(). Source: "
        "LangGraph blog post (blog.langchain.dev/langgraph/).",
    }
    report = synthesise_report(
        "What are the main differences between PydanticAI and LangGraph?",
        "analytical",
        mock_evidence,
    )
    print(report.model_dump_json(indent=2))
