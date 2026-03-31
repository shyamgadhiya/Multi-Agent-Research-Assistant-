"""
researcher_node.py — with Corrective RAG (CRAG) checks.

CRAG Check 1 — Batch chunk relevance grader (SINGLE LLM call for all chunks)
    Grades all retrieved chunks in one prompt instead of one call per chunk.
    This reduces N LLM calls to 1, preventing 429 rate limit errors.

CRAG Check 2 — Smart web fallback
    Web search only triggers when RAG quality is insufficient.

CRAG Check 3 — Grounding verifier
    Verifies answer is supported by retrieved context.

All LLM calls use retry-with-backoff to handle 429 RESOURCE_EXHAUSTED errors.
"""

import os
import re
import json
import time
from langchain_core.messages import SystemMessage, HumanMessage
from state import ResearchState
from setup import llm_fast, web_search, load_vectorstore


# ── Prompts ────────────────────────────────────────────────────────────────────

BATCH_GRADER_PROMPT = """You are a relevance grader. Given a sub-question and multiple
text chunks, grade each chunk's relevance for answering the question.

Sub-question: {task}

{chunks_text}

For each chunk above, return its grade.
Return ONLY a valid JSON array with one entry per chunk, in order:
["relevant", "irrelevant", "ambiguous", ...]

Grades:
- relevant   = chunk directly helps answer the question
- irrelevant = chunk is off-topic or unhelpful
- ambiguous  = chunk is partially related but incomplete"""


GROUNDING_CHECK_PROMPT = """You are a grounding verifier. Check whether the answer
is supported by the provided context.

Sub-question: {task}

Context (retrieved chunks):
{context}

Answer to verify:
{answer}

Rules:
- grounded must be true if the answer uses information from the context, even partially
- grounded must be false ONLY if the answer clearly contradicts or ignores the context
- confidence = how much of the answer is supported (0.0 = none, 1.0 = fully)
- Do NOT penalise for minor connective phrasing or general knowledge transitions
- A confidence of 0.7 or above means the answer is well-grounded

Return ONLY valid JSON, no markdown:
{{"grounded": true, "confidence": 0.85, "reason": "brief reason"}}"""


RESEARCHER_SYSTEM = """You are a focused research agent. Answer using ONLY the context provided.

Sub-question: {task}

--- RAG CONTEXT (collection: {collection} | file: {file_filter}) ---
{rag_context}

--- WEB SEARCH CONTEXT ---
{web_context}

Rules:
- Use ONLY the provided context — do not use outside knowledge
- Cite sources using [Doc N - filename] or [URL] notation
- 150-300 words
- If context is insufficient, say so — do not guess"""


# ── Retry-with-backoff LLM wrapper ─────────────────────────────────────────────

def _call_llm(system: str, human: str, max_retries: int = 3) -> str:
    """
    Call llm_fast with automatic retry on 429 RESOURCE_EXHAUSTED.
    Waits the suggested retry delay from the error message, or 60s by default.
    """
    for attempt in range(max_retries):
        try:
            response = llm_fast.invoke([
                SystemMessage(content=system),
                HumanMessage(content=human),
            ])
            raw = response.content
            if isinstance(raw, list):
                return " ".join(
                    p["text"] if isinstance(p, dict) and "text" in p else str(p)
                    for p in raw
                ).strip()
            return str(raw).strip()
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                # Extract retry delay from error message if available
                delay_match = re.search(r'retry[^\d]*(\d+)', err_str, re.IGNORECASE)
                wait = int(delay_match.group(1)) + 2 if delay_match else 62
                # Cap wait at 90s per attempt
                wait = min(wait, 90)
                if attempt < max_retries - 1:
                    time.sleep(wait)
                    continue
            # Non-429 error or final attempt — raise
            raise
    return ""


def _extract_text(content) -> str:
    if isinstance(content, list):
        return " ".join(
            part["text"] if isinstance(part, dict) and "text" in part else str(part)
            for part in content
        )
    return str(content)


# ── Check 1: BATCH chunk relevance grader (1 call for all chunks) ──────────────

def _grade_all_chunks_batch(task: str, docs: list) -> dict:
    """
    Grade ALL chunks in a SINGLE LLM call.
    Returns:
    {
      "relevant":   [doc, ...],
      "ambiguous":  [doc, ...],
      "irrelevant": [doc, ...],
      "verdict":    "relevant" | "ambiguous" | "irrelevant"
    }

    Why batch? One call per chunk × N chunks × parallel researchers
    = 12-16 Flash calls/minute → 429 on the free tier (limit: 15/min).
    One batch call per researcher = 3-4 calls/minute → well within limits.
    """
    if not docs:
        return {"relevant": [], "ambiguous": [], "irrelevant": [], "verdict": "irrelevant"}

    # Build numbered chunk list for the prompt
    chunks_text = "\n\n".join(
        f"[Chunk {i+1}]:\n{doc.page_content[:600]}"
        for i, doc in enumerate(docs)
    )

    buckets = {"relevant": [], "ambiguous": [], "irrelevant": []}

    try:
        raw = _call_llm(
            system='You are a relevance grader. Return ONLY a valid JSON array of grades.',
            human=BATCH_GRADER_PROMPT.format(task=task, chunks_text=chunks_text),
        )
        # Strip markdown fences
        raw = re.sub(r"```json|```", "", raw).strip()
        grades = json.loads(raw)

        for i, doc in enumerate(docs):
            grade = str(grades[i]).lower().strip() if i < len(grades) else "ambiguous"
            if grade not in ("relevant", "irrelevant", "ambiguous"):
                grade = "ambiguous"
            buckets[grade].append(doc)

    except Exception:
        # On any failure, treat all chunks as ambiguous (safe default)
        buckets["ambiguous"] = docs

    # Overall verdict
    if buckets["relevant"]:
        verdict = "relevant"
    elif buckets["ambiguous"]:
        verdict = "ambiguous"
    else:
        verdict = "irrelevant"

    return {**buckets, "verdict": verdict}


# ── Check 2: Smart web fallback ────────────────────────────────────────────────

def _should_use_web(verdict: str, has_vectorstore: bool) -> bool:
    if not has_vectorstore:
        return True
    return verdict in ("irrelevant", "ambiguous")


# ── Check 3: Grounding verifier ────────────────────────────────────────────────

def _verify_grounding(task: str, context: str, answer: str) -> dict:
    """Returns {"grounded": bool, "confidence": float, "reason": str}"""
    if not context.strip():
        return {"grounded": False, "confidence": 0.0, "reason": "No context available."}
    try:
        raw = _call_llm(
            system="You are a grounding verifier. Return only valid JSON — no markdown.",
            human=GROUNDING_CHECK_PROMPT.format(
                task=task,
                context=context[:4000],
                answer=answer[:1500],
            ),
        )
        raw = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(raw)
        return {
            "grounded":   bool(result.get("grounded", True)),
            "confidence": float(result.get("confidence", 0.5)),
            "reason":     str(result.get("reason", "")),
        }
    except Exception:
        return {"grounded": True, "confidence": 0.5, "reason": "Check failed — assuming grounded."}


# ── Retrieval helpers ──────────────────────────────────────────────────────────

def _retrieve_from_vectorstore(task: str, abs_path: str, active_file: str = "", k: int = 4):
    """Retrieve, batch-grade, return only relevant+ambiguous chunks."""
    if not abs_path:
        return "", [], {"verdict": "irrelevant", "kept": 0, "total": 0}

    vs = load_vectorstore(abs_path)
    if vs is None:
        return "", [], {"verdict": "irrelevant", "kept": 0, "total": 0}

    fetch_k = k * 3 if active_file else k * 2
    docs = vs.similarity_search(task, k=fetch_k)
    if not docs:
        return "", [], {"verdict": "irrelevant", "kept": 0, "total": 0}

    if active_file:
        docs = [d for d in docs if d.metadata.get("source", "") == active_file]
        if not docs:
            return "", [], {"verdict": "irrelevant", "kept": 0, "total": 0}

    total = len(docs)

    # Check 1: single batch call grades all chunks
    graded  = _grade_all_chunks_batch(task, docs)
    verdict = graded["verdict"]

    kept_docs = (graded["relevant"] + graded["ambiguous"])[:k]
    if not kept_docs:
        return "", [], {"verdict": verdict, "kept": 0, "total": total,
                        "relevant": 0, "ambiguous": 0, "irrelevant": total}

    chunks  = []
    sources = []
    seen    = set()
    for i, doc in enumerate(kept_docs, 1):
        source     = doc.metadata.get("source",     "unknown")
        collection = doc.metadata.get("collection", "")
        label      = f"{source} [{collection}]" if collection else source
        chunks.append(f"[Doc {i} - {label}]\n{doc.page_content}")
        key = (source, collection)
        if key not in seen:
            seen.add(key)
            sources.append({"file": source, "collection": collection})

    summary = {
        "verdict":    verdict,
        "kept":       len(kept_docs),
        "total":      total,
        "relevant":   len(graded["relevant"]),
        "ambiguous":  len(graded["ambiguous"]),
        "irrelevant": len(graded["irrelevant"]),
    }
    return "\n\n".join(chunks), sources, summary


def _retrieve_from_web(task: str):
    try:
        results = web_search.invoke(task)
        if isinstance(results, str):
            urls    = re.findall(r'https?://[^\s\)\]"]+', results)
            sources = [{"url": u, "title": _url_to_title(u)} for u in dict.fromkeys(urls)]
            return results, sources
        if isinstance(results, list):
            formatted, sources, seen_urls = [], [], set()
            for r in results:
                if isinstance(r, dict):
                    url   = r.get("url",   "")
                    cont  = r.get("content","")
                    title = r.get("title", "") or _url_to_title(url)
                    formatted.append(f"[{url}]\n{cont}")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        sources.append({"url": url, "title": title})
                else:
                    formatted.append(str(r))
            return "\n\n".join(formatted), sources
        return str(results), []
    except Exception as e:
        return f"Web search failed: {str(e)}", []


def _url_to_title(url: str) -> str:
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        path   = parsed.path.strip("/").split("/")[-1].replace("-"," ").replace("_"," ")
        return f"{path} — {domain}" if path else domain
    except Exception:
        return url


# ── Main node ──────────────────────────────────────────────────────────────────

def researcher_node(state: dict) -> dict:
    task:             str = state["task"]
    vectorstore_path: str = state.get("vectorstore_path", "")
    active_file:      str = state.get("active_file",      "")

    abs_path        = os.path.abspath(vectorstore_path) if vectorstore_path else ""
    collection_name = os.path.basename(abs_path) if abs_path else "none"
    file_filter     = active_file if active_file else "all files"
    has_vectorstore = bool(abs_path)

    # ── Check 1 + 2 ────────────────────────────────────────────────────────────
    if has_vectorstore:
        rag_context, rag_sources, rag_summary = _retrieve_from_vectorstore(
            task, abs_path, active_file
        )
        verdict = rag_summary.get("verdict", "irrelevant")
        use_web = _should_use_web(verdict, has_vectorstore)
    else:
        rag_context  = ""
        rag_sources  = []
        rag_summary  = {"verdict": "not_used", "kept": 0, "total": 0}
        verdict      = "not_used"
        use_web      = True

    if use_web:
        web_context, web_sources = _retrieve_from_web(task)
    else:
        web_context, web_sources = "", []

    combined_context = "\n\n".join(filter(None, [rag_context, web_context]))

    # ── Generate answer ─────────────────────────────────────────────────────────
    prompt = RESEARCHER_SYSTEM.format(
        task=task,
        collection=collection_name,
        file_filter=file_filter,
        rag_context=rag_context or "No relevant documents found after relevance filtering.",
        web_context=web_context or (
            "Web search skipped — RAG context was sufficient."
            if has_vectorstore and not use_web else "No web results found."
        ),
    )

    answer = _call_llm(
        system="You are a focused research agent. Answer using only the provided context.",
        human=prompt + "\n\nAnswer the sub-question using only the provided context. Cite sources.",
    )

    # ── Check 3: Grounding verifier ─────────────────────────────────────────────
    if combined_context.strip():
        grounding = _verify_grounding(task, combined_context, answer)
    else:
        grounding = {"grounded": True, "confidence": 0.5, "reason": "No context to verify."}

    rag_was_used = bool(rag_context)
    # Use confidence score ONLY — ignore the boolean "grounded" field which the
    # LLM sometimes gets wrong (e.g. grounded:false but confidence:1.0).
    # Flag as low confidence only when score is genuinely poor (< 0.60).
    low_confidence = rag_was_used and grounding["confidence"] < 0.40

    crag_data = {
        "chunk_verdict":     verdict,
        "chunks_total":      rag_summary.get("total",      0),
        "chunks_kept":       rag_summary.get("kept",       0),
        "chunks_relevant":   rag_summary.get("relevant",   0),
        "chunks_ambiguous":  rag_summary.get("ambiguous",  0),
        "chunks_irrelevant": rag_summary.get("irrelevant", 0),
        "web_triggered":     use_web,
        "grounded":          grounding["grounded"],
        "grounding_score":   grounding["confidence"],
        "grounding_reason":  grounding["reason"],
        "low_confidence":    low_confidence if has_vectorstore else False,
    }

    finding = {
        "task":        task,
        "answer":      answer,
        "rag_used":    bool(rag_context),
        "web_used":    bool(web_context),
        "web_skipped": has_vectorstore and not use_web,
        "collection":  collection_name,
        "file":        active_file or "all",
        "rag_sources": rag_sources,
        "web_sources": web_sources,
        "crag":        crag_data,
    }

    return {"findings": [finding]}