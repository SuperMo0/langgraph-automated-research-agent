from typing import Any, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from core.config import MAX_SYNTHESIS_ITERATIONS
from core.models import EvaluationResult, EvidenceStore, QueryType, ResearchReport, RouterDecision
from pipeline.evaluator import evaluate_report
from pipeline.research_agent import run_research_agent
from pipeline.router import classify_query
from pipeline.synthesis import synthesise_report


class PipelineState(TypedDict):
    query: str
    verbose: bool
    query_type: NotRequired[QueryType]
    router_decision: NotRequired[RouterDecision]
    evidence: NotRequired[EvidenceStore]
    report: NotRequired[ResearchReport]
    evaluation: NotRequired[EvaluationResult]
    iteration: NotRequired[int]


def _require_query_type(state: PipelineState) -> QueryType:
    query_type = state.get("query_type")
    if query_type is None:
        raise RuntimeError("node reached without a query_type in state")
    return query_type


def route_node(state: PipelineState) -> dict[str, Any]:
    decision = classify_query(state["query"])
    print(f"[Router] Query type: {decision.query_type} (confidence: {decision.confidence})")
    return {"query_type": decision.query_type, "router_decision": decision}


def research_node(state: PipelineState) -> dict[str, Any]:
    print("[Research] Starting information gathering...")
    evidence = run_research_agent(state["query"], _require_query_type(state), verbose=state["verbose"])
    return {"evidence": evidence}


def synthesis_node(state: PipelineState) -> dict[str, Any]:
    previous_report = state.get("report")
    evaluation = state.get("evaluation")
    if previous_report is not None and evaluation is not None:
        print(f"[Synthesis] Retry with feedback: {evaluation.feedback!r}")
    else:
        print("[Synthesis] Drafting report...")

    report = synthesise_report(
        state["query"],
        _require_query_type(state),
        state.get("evidence", {}),
        previous_report=previous_report,
        feedback=evaluation.feedback if evaluation is not None else None,
    )
    return {"report": report, "iteration": state.get("iteration", 0) + 1}


def evaluation_node(state: PipelineState) -> dict[str, Any]:
    report = state.get("report")
    if report is None:
        raise RuntimeError("evaluation_node reached without a report in state")

    evaluation = evaluate_report(report, state.get("evidence", {}))
    mark = "✓" if evaluation.passed else "✗"
    print(
        f"[Evaluator] Scores: completeness={evaluation.completeness}, "
        f"accuracy={evaluation.accuracy}, clarity={evaluation.clarity}, "
        f"sourcing={evaluation.sourcing} → avg {evaluation.average:.2f} {mark}"
    )
    return {"evaluation": evaluation}


def should_continue(state: PipelineState) -> str:
    evaluation = state.get("evaluation")
    if evaluation is not None and evaluation.passed:
        return END
    if state.get("iteration", 0) >= MAX_SYNTHESIS_ITERATIONS:
        return END
    return "synthesis"


def build_graph() -> CompiledStateGraph[PipelineState, None, PipelineState, PipelineState]:
    graph = StateGraph(PipelineState)
    graph.add_node("router", route_node)
    graph.add_node("research", research_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("evaluation", evaluation_node)

    graph.add_edge(START, "router")
    graph.add_edge("router", "research")
    graph.add_edge("research", "synthesis")
    graph.add_edge("synthesis", "evaluation")
    graph.add_conditional_edges("evaluation", should_continue, {"synthesis": "synthesis", END: END})

    return graph.compile()


def run_pipeline(query: str, *, verbose: bool = False) -> ResearchReport:
    graph = build_graph()
    final_state = graph.invoke({"query": query, "verbose": verbose})
    report = final_state.get("report")
    if report is None:
        raise RuntimeError("Pipeline finished without producing a report")
    return report


if __name__ == "__main__":
    import sys

    test_query = sys.argv[1] if len(sys.argv) > 1 else "How does vector search work?"
    final_report = run_pipeline(test_query, verbose=True)
    print(f"\n[Report] query_type={final_report.query_type} confidence={final_report.confidence}")
    print(final_report.model_dump_json(indent=2))
