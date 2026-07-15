from models import EvidenceStore


def format_evidence(evidence: EvidenceStore) -> str:
    if not evidence:
        return "(no evidence was gathered)"
    return "\n\n".join(f"[{key}]\n{content}" for key, content in evidence.items())
