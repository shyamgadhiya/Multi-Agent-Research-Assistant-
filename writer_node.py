from langchain_core.messages import SystemMessage, HumanMessage
from state import ResearchState
from setup import llm

WRITER_SYSTEM = """You are an expert research writer. Synthesize the research findings
below into a comprehensive, well-structured report.

Requirements:
- Start with an executive summary (2-3 sentences)
- Use clear markdown headings (##, ###)
- Cite web sources as proper markdown links: [Source Name](https://url.com)
  NEVER write raw URLs like https://... or [https://...] directly in the text
  ALWAYS use the format [descriptive title](url) for every web citation
- Cite document sources as [filename.pdf]
- Include a "Key Takeaways" section at the end with 3-5 bullet points
- Aim for 500-800 words
- Be factual, balanced, and clear

Original query: {query}

--- RESEARCH FINDINGS ---
{findings_text}
"""


def _extract_text(content) -> str:
    if isinstance(content, list):
        return " ".join(
            part["text"] if isinstance(part, dict) and "text" in part else str(part)
            for part in content
        )
    return str(content)


def writer_node(state: ResearchState) -> dict:
    findings      = state.get("findings", [])
    findings_text = _format_findings(findings)

    prompt = WRITER_SYSTEM.format(
        query=state["query"],
        findings_text=findings_text,
    )

    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Write the final research report now."),
    ])

    report_body = _extract_text(response.content)

    # Append a structured Resources section compiled from all findings
    resources_section = _build_resources_section(findings)
    if resources_section:
        final_report = f"{report_body.rstrip()}\n\n{resources_section}"
    else:
        final_report = report_body

    return {"final_report": final_report}


def _format_findings(findings: list[dict]) -> str:
    if not findings:
        return "No findings available."

    sections = []
    for i, f in enumerate(findings, 1):
        task         = f.get("task", f"Sub-task {i}")
        answer       = f.get("answer", "No answer provided.")
        rag          = "RAG" if f.get("rag_used") else ""
        web          = "Web" if f.get("web_used") else ""
        sources_used = ", ".join(filter(None, [rag, web])) or "none"
        sections.append(
            f"### Sub-task {i}: {task}\n"
            f"Sources used: {sources_used}\n\n"
            f"{answer}"
        )
    return "\n\n".join(sections)


def _build_resources_section(findings: list[dict]) -> str:
    """
    Collect unique RAG files and web URLs across all findings
    and format them as a markdown Resources section.
    """
    rag_seen  = {}   # key=(file, collection) → dict
    web_seen  = {}   # key=url → dict

    for f in findings:
        for rs in f.get("rag_sources", []):
            key = (rs.get("file", ""), rs.get("collection", ""))
            if key not in rag_seen:
                rag_seen[key] = rs

        for ws in f.get("web_sources", []):
            url = ws.get("url", "").strip()
            if url and url not in web_seen:
                web_seen[url] = ws

    if not rag_seen and not web_seen:
        return ""

    lines = ["---", "## 📚 Resources"]

    if rag_seen:
        lines.append("\n### Documents (RAG)")
        for (fname, col), _ in rag_seen.items():
            col_label = f" · *{col}*" if col and col != "none" else ""
            lines.append(f"- 📄 **{fname}**{col_label}")

    if web_seen:
        lines.append("\n### Web Sources")
        for url, ws in web_seen.items():
            title = ws.get("title", "") or url
            lines.append(f"- 🌐 [{title}]({url})")

    return "\n".join(lines)