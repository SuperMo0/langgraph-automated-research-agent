import re
from pathlib import Path

from core.models import ResearchReport

_DIVIDER = "═" * 48


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60].rstrip("-")


def _normalize_url(url: str) -> str:
    return url if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url) else f"https://{url}"


def render_markdown(report: ResearchReport) -> str:
    sources = [
        f"{i}. [{s.title}]({_normalize_url(s.url)}) — {s.relevance}" if s.url else f"{i}. {s.title} — {s.relevance}"
        for i, s in enumerate(report.sources, start=1)
    ]
    lines = [
        "# Research Report",
        "",
        f"**Query:** {report.query}  ",
        f"**Type:** {report.query_type} | **Confidence:** {report.confidence}  ",
        f"**Generated:** {report.generated_at}",
        "",
        "## Summary",
        "",
        report.summary,
        "",
        "## Key Findings",
        "",
        *(f"- {finding}" for finding in report.key_findings),
        "",
        "## Sources",
        "",
        *sources,
        "",
        "## Limitations",
        "",
        report.limitations,
        "",
    ]
    return "\n".join(lines)


def render_console(report: ResearchReport) -> str:
    sources = [
        f"{i}. {s.title}" + (f" ({_normalize_url(s.url)})" if s.url else "")
        for i, s in enumerate(report.sources, start=1)
    ]
    lines = [
        _DIVIDER,
        "RESEARCH REPORT",
        _DIVIDER,
        f"Query: {report.query}",
        f"Type:  {report.query_type} | Confidence: {report.confidence}",
        "",
        "SUMMARY",
        report.summary,
        "",
        "KEY FINDINGS",
        *(f"• {finding}" for finding in report.key_findings),
        "",
        "SOURCES",
        *sources,
        "",
        "LIMITATIONS",
        report.limitations,
        _DIVIDER,
    ]
    return "\n".join(lines)


def save_report(report: ResearchReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f"{report.generated_at[:10]}-{slugify(report.query)}"
    md_path = base.with_suffix(".md")
    json_path = base.with_suffix(".json")
    md_path.write_text(render_markdown(report))
    json_path.write_text(report.model_dump_json(indent=2))
    return md_path, json_path
