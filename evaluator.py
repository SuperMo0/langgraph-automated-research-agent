from openai.types.chat import ChatCompletionMessageParam

from evidence import format_evidence
from llm_client import parse_chat_completion
from models import EvaluationResult, EvidenceStore, ResearchReport

_SYSTEM_PROMPT = """You are a strict evaluator of research reports. Score the given report \
on four criteria, each from 1 (very poor) to 10 (excellent):

- completeness: does it fully answer the original query?
- accuracy: are the claims well-supported by the provided evidence? Penalise claims that \
go beyond what the evidence actually supports.
- clarity: is it well-structured, readable, and free of redundancy?
- sourcing: does it cite specific, identifiable sources for its key claims?

Be a strict, honest grader — do not default to high scores. In `feedback`, give concrete, \
actionable guidance for what to change if any score is below 8, referencing specific \
findings or sections. If every score is already 8 or above, feedback should simply confirm \
the report is solid."""


def _build_messages(report: ResearchReport, evidence: EvidenceStore) -> list[ChatCompletionMessageParam]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Original query: {report.query}\n\n"
            f"Evidence available to the report writer:\n{format_evidence(evidence)}\n\n"
            f"Report to evaluate:\n{report.model_dump_json(indent=2)}",
        },
    ]


def evaluate_report(report: ResearchReport, evidence: EvidenceStore) -> EvaluationResult:
    messages = _build_messages(report, evidence)
    response = parse_chat_completion(messages, EvaluationResult)
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("Evaluator model did not return a structured evaluation.")
    return parsed


if __name__ == "__main__":
    from synthesis import synthesise_report

    mock_evidence: EvidenceStore = {
        "pydanticai_typing": "PydanticAI enforces typed result models via Pydantic, so "
        "agents cannot return unvalidated output. Source: PydanticAI docs "
        "(ai.pydantic.dev/agents/).",
        "langgraph_state": "LangGraph models workflows as explicit state graphs with visible "
        "nodes and edges, and has native human-in-the-loop support via interrupt(). Source: "
        "LangGraph blog post (blog.langchain.dev/langgraph/).",
    }
    mock_report = synthesise_report(
        "What are the main differences between PydanticAI and LangGraph?",
        "analytical",
        mock_evidence,
    )
    result = evaluate_report(mock_report, mock_evidence)
    print(result.model_dump_json(indent=2))
    print(f"average={result.average} passed={result.passed}")
