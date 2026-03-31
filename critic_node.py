import json
from langchain_core.messages import SystemMessage, HumanMessage
from state import ResearchState
from setup import llm

CRITIC_SYSTEM = """You are a research quality critic. Evaluate the research findings
below against the original query.

IMPORTANT CONTEXT:
- Research mode: {research_mode}
- Topic difficulty: {topic_difficulty}
- Retry number: {retry_count}

Scoring guidelines based on context:
- For WEB-ONLY research on niche/specialist topics (NGOs, policy, social science,
  conflict studies): a score of 0.65-0.80 is GOOD if findings cover the main angles
  with credible sources, even if exact statistics are unavailable
- For RAG research over private documents: expect higher precision, score 0.75-0.95
- Do NOT penalise for lack of specific statistics if the topic is inherently data-sparse
- Do NOT penalise for "insufficient depth" on a first pass — depth improves on retry
- Only mark a gap if a RETRY could realistically find better information

Score the findings from 0.0 to 1.0 on:
- Coverage: Are the main angles of the query addressed?
- Source quality: Are claims backed by credible sources?
- Usefulness: Is there enough to write a meaningful report?

Return ONLY valid JSON in this exact format:
{{
  "score": 0.85,
  "gaps": ["gap 1 that a retry could fix", "gap 2"],
  "reasoning": "One sentence summary."
}}

Original query: {query}

--- FINDINGS ---
{findings_text}
"""

# Topics that are inherently hard to find precise data on —
# lower the effective threshold for these
SPECIALIST_KEYWORDS = {
    "ngo", "peacebuilding", "non-violence", "conflict resolution",
    "humanitarian", "civil society", "advocacy", "governance",
    "kpi", "m&e", "monitoring", "evaluation", "sustainability",
    "qualitative", "policy", "diplomacy", "grassroots",
}

PASS_THRESHOLD = 0.75
MAX_RETRIES    = 2


def _extract_text(content) -> str:
    if isinstance(content, list):
        return " ".join(
            part["text"] if isinstance(part, dict) and "text" in part else str(part)
            for part in content
        )
    return str(content)


def _detect_topic_difficulty(query: str, findings: list[dict]) -> str:
    """
    Returns 'specialist' or 'general'.
    Specialist topics get a more lenient effective threshold.
    """
    q_lower = query.lower()
    if any(kw in q_lower for kw in SPECIALIST_KEYWORDS):
        return "specialist"
    # Also flag as specialist if web search returned sparse results
    avg_answer_len = sum(len(f.get("answer", "")) for f in findings) / max(len(findings), 1)
    if avg_answer_len < 300:
        return "specialist"
    return "general"


def _effective_threshold(topic_difficulty: str, retry_count: int) -> float:
    """
    Adjust pass threshold based on topic difficulty and retry count.
    - Specialist topics: lower base threshold (0.60)
    - Each retry that doesn't improve: relax threshold slightly
    """
    base = 0.60 if topic_difficulty == "specialist" else PASS_THRESHOLD
    # After first retry with no improvement, relax by 0.05
    relaxation = min(retry_count * 0.05, 0.10)
    return max(base - relaxation, 0.50)   # never below 0.50


def critic_node(state: ResearchState) -> dict:
    findings     = state.get("findings", [])
    retry_count  = state.get("retry_count", 0)
    query        = state.get("query", "")

    # Detect whether RAG or web was used
    has_rag  = any(f.get("rag_used") for f in findings)
    has_web  = any(f.get("web_used") for f in findings)
    research_mode = (
        "RAG + Web (hybrid)" if has_rag and has_web
        else "RAG only"      if has_rag
        else "Web only"
    )

    topic_difficulty = _detect_topic_difficulty(query, findings)
    findings_text    = _format_findings(findings)

    prompt = CRITIC_SYSTEM.format(
        query=query,
        findings_text=findings_text,
        research_mode=research_mode,
        topic_difficulty=topic_difficulty,
        retry_count=retry_count,
    )

    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Evaluate the research quality now."),
    ])

    raw = _extract_text(response.content).strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    result    = json.loads(raw)
    llm_score = float(result["score"])
    gaps      = result.get("gaps", [])

    # CRAG penalty — only apply when RAG was used AND confidence is genuinely poor.
    # Penalty is small (0.05 per finding, max 0.15) so a single weak answer
    # doesn't tank an otherwise good run.
    if has_rag:
        low_confidence_count = sum(
            1 for f in findings
            if f.get("crag", {}).get("low_confidence", False)
            and f.get("crag", {}).get("grounding_score", 1.0) < 0.50
        )
        crag_penalty = min(low_confidence_count * 0.05, 0.15)
    else:
        crag_penalty = 0.0

    final_score = max(0.0, round(llm_score - crag_penalty, 3))

    # Add low-confidence RAG tasks to gaps
    if has_rag:
        for f in findings:
            crag = f.get("crag", {})
            if crag.get("low_confidence") and f.get("task"):
                gap_entry = f"Low-confidence answer: {f['task'][:80]}"
                if gap_entry not in gaps:
                    gaps.append(gap_entry)

    return {
        "critic_score":       final_score,
        "gaps":               gaps,
        # Store for routing decision
        "_topic_difficulty":  topic_difficulty,
        "_research_mode":     research_mode,
    }


def route_after_critic(state: ResearchState) -> str:
    score            = state.get("critic_score",      0.0)
    retry_count      = state.get("retry_count",       0)
    topic_difficulty = state.get("_topic_difficulty", "general")

    # Hard cap — never retry more than MAX_RETRIES times
    if retry_count >= MAX_RETRIES:
        return "writer_node"

    # Use context-aware threshold instead of fixed 0.75
    threshold = _effective_threshold(topic_difficulty, retry_count)

    if score >= threshold:
        return "writer_node"
    return "planner_node"


def _format_findings(findings: list[dict]) -> str:
    if not findings:
        return "No findings available."

    sections = []
    for i, f in enumerate(findings, 1):
        task   = f.get("task",   f"Sub-task {i}")
        answer = f.get("answer", "No answer provided.")
        crag   = f.get("crag",   {})

        crag_note = ""
        if crag and crag.get("chunk_verdict") != "not_used":
            verdict  = crag.get("chunk_verdict",   "unknown")
            kept     = crag.get("chunks_kept",     0)
            total    = crag.get("chunks_total",    0)
            grounded = crag.get("grounded",        True)
            g_score  = crag.get("grounding_score", 1.0)
            web_trig = crag.get("web_triggered",   False)
            crag_note = (
                f"\n[CRAG] RAG verdict: {verdict} | chunks: {kept}/{total} kept | "
                f"web triggered: {web_trig} | grounded: {grounded} ({g_score:.0%})"
            )

        sections.append(f"### Sub-task {i}: {task}{crag_note}\n{answer}")

    return "\n\n".join(sections)