from typing import TypedDict, Annotated


def _append_findings(left: list, right: list) -> list:
    """Simple append reducer — merges parallel researcher outputs into one list."""
    return left + right


class ResearchState(TypedDict):
    # Input
    query:            str
    vectorstore_path: str
    active_file:      str
    images_enabled:   bool

    # Planner output
    sub_tasks: list[str]

    # Parallel researcher outputs — custom append reducer (NOT add_messages)
    findings: Annotated[list[dict], _append_findings]

    # Critic output
    critic_score:      float
    gaps:              list[str]
    retry_count:       int

    # Critic metadata — used by route_after_critic for context-aware threshold
    # These MUST be in state so LangGraph doesn't silently drop them
    _topic_difficulty: str   # "specialist" | "general"
    _research_mode:    str   # "Web only" | "RAG only" | "RAG + Web (hybrid)"

    # Writer output
    final_report: str

    # Image node output
    image_insertions: list[dict]