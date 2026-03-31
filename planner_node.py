import json
from langchain_core.messages import SystemMessage, HumanMessage
from state import ResearchState
from setup import llm
from langgraph.types import Send

PLANNER_SYSTEM = """You are a research planner. Your job is to break a user's query
into 3-5 focused sub-questions that can be researched independently and in parallel.

{gaps_instruction}

Rules:
- Each sub-question must be self-contained and specific
- Cover different angles: background, current state, implications, examples
- Do NOT overlap between sub-questions

Return ONLY a valid JSON array of strings, no markdown, no explanation.

Example output:
["What is quantum entanglement?", "How does quantum computing threaten RSA encryption?", "What post-quantum cryptography standards exist in 2025?"]"""


def _extract_text(content) -> str:
    if isinstance(content, list):
        return " ".join(
            part["text"] if isinstance(part, dict) and "text" in part else str(part)
            for part in content
        )
    return str(content)


def planner_node(state: ResearchState) -> dict:
    gaps = state.get("gaps", [])

    if gaps:
        gaps_instruction = (
            f"IMPORTANT: The previous research attempt was graded insufficient. "
            f"These topics were missing or poorly covered: {gaps}. "
            f"Make sure your sub-questions explicitly target these gaps."
        )
    else:
        gaps_instruction = ""

    prompt = PLANNER_SYSTEM.format(gaps_instruction=gaps_instruction)

    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"Research query: {state['query']}"),
    ])

    raw = _extract_text(response.content).strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    sub_tasks: list[str] = json.loads(raw)

    return {
        "sub_tasks":   sub_tasks,
        "retry_count": state.get("retry_count", 0) + (1 if gaps else 0),
    }


def route_to_researchers(state: ResearchState) -> list[Send]:
    return [
        Send("researcher_node", {
            "task":             task,
            "query":            state["query"],
            "vectorstore_path": state.get("vectorstore_path", ""),
            "active_file":      state.get("active_file", ""),   # pass file filter
        })
        for task in state["sub_tasks"]
    ]