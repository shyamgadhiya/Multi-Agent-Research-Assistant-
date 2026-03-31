"""
image_node — runs after writer_node when images_enabled=True.

Image sources (tried in order):
  1. Tavily search with "images" topic — real web images, works for current events
  2. Wikimedia Commons API — reliable for encyclopedic/scientific topics
  3. Unsplash Source — generic high-quality fallback photo

No new API key needed — reuses your existing TAVILY_API_KEY.
"""

import re
import os
import time
import json
import urllib.parse
import urllib.request
from state import ResearchState
from setup import llm
from langchain_core.messages import SystemMessage, HumanMessage


# ── Source 1: Tavily image search ─────────────────────────────────────────────

def _tavily_image_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Use Tavily search API with include_images=True.
    Reuses TAVILY_API_KEY from environment — no extra cost beyond normal search quota.
    Returns list of {"url": str, "title": str, "source": str}
    """
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return []

    try:
        payload = json.dumps({
            "api_key":        api_key,
            "query":          query,
            "include_images": True,
            "max_results":    max_results,
            "search_depth":   "basic",
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent":   "ResearchAssistant/1.0",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        images  = data.get("images", [])
        results = data.get("results", [])

        # images is a list of URLs in newer Tavily versions
        # pair each image URL with the closest result for title/source
        output = []
        for i, img in enumerate(images[:max_results]):
            # img can be a plain URL string or a dict
            if isinstance(img, dict):
                url   = img.get("url", "")
                title = img.get("title", query)
            else:
                url   = str(img)
                title = results[i]["title"] if i < len(results) else query

            if not url or not url.startswith("http"):
                continue
            # skip SVGs and tiny data URIs
            if url.endswith(".svg") or url.startswith("data:"):
                continue

            source = results[i]["url"] if i < len(results) else url
            output.append({"url": url, "title": title[:120], "source": source})

        return output

    except Exception:
        return []


# ── Source 2: Wikimedia Commons (good for science/history topics) ─────────────

def _wikimedia_image_search(query: str, max_results: int = 3) -> list[dict]:
    """Search Wikimedia Commons for freely licensed images."""
    try:
        params = urllib.parse.urlencode({
            "action":       "query",
            "generator":    "search",
            "gsrnamespace": "6",
            "gsrsearch":    f"filetype:bitmap {query}",
            "gsrlimit":     str(max_results * 2),
            "prop":         "imageinfo",
            "iiprop":       "url|size|mime",
            "iiurlwidth":   "800",
            "format":       "json",
            "origin":       "*",
        })
        url = f"https://commons.wikimedia.org/w/api.php?{params}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ResearchAssistant/1.0 (educational use)"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))

        pages   = data.get("query", {}).get("pages", {})
        results = []
        for page in pages.values():
            info_list = page.get("imageinfo", [])
            if not info_list:
                continue
            info = info_list[0]
            mime = info.get("mime", "")
            if not mime.startswith("image/") or "svg" in mime:
                continue
            w = info.get("width",  0)
            h = info.get("height", 0)
            if w < 300 or h < 200:
                continue
            img_url = info.get("url", "")
            if not img_url:
                continue
            title  = page.get("title", query).replace("File:", "").rsplit(".", 1)[0]
            source = info.get("descriptionurl", "https://commons.wikimedia.org")
            results.append({"url": img_url, "title": title[:120], "source": source})
            if len(results) >= max_results:
                break

        return results
    except Exception:
        return []


# ── Source 3: Unsplash fallback ────────────────────────────────────────────────

def _unsplash_fallback(query: str) -> list[dict]:
    """Random relevant Unsplash photo — no API key needed."""
    try:
        encoded = urllib.parse.quote(query.replace(" ", ","))
        url = f"https://source.unsplash.com/800x500/?{encoded}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ResearchAssistant/1.0"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            ct        = resp.headers.get("Content-Type", "")
            final_url = resp.url
            if "image" in ct and final_url and "photo" in final_url:
                return [{"url": final_url,
                         "title": f"{query}",
                         "source": "https://unsplash.com"}]
    except Exception:
        pass
    return []


# ── Unified search: tries all sources in order ────────────────────────────────

def _search_image(query: str) -> dict | None:
    """
    Try Tavily → Wikimedia → Unsplash.
    Returns the first valid image dict or None.
    """
    # 1. Tavily (best for current events, news, conflict topics)
    results = _tavily_image_search(query, max_results=5)
    if results:
        return results[0]

    # 2. Wikimedia (best for science, history, geography)
    results = _wikimedia_image_search(query, max_results=3)
    if results:
        return results[0]

    # 3. Unsplash (generic fallback)
    results = _unsplash_fallback(query)
    if results:
        return results[0]

    return None


# ── Query generation ───────────────────────────────────────────────────────────

def _generate_image_query(section_title: str, topic: str) -> str:
    """Use LLM to produce a concrete 4-7 word image search query."""
    try:
        response = llm.invoke([
            SystemMessage(content=(
                "You produce short, specific image search queries (4-7 words). "
                "Return ONLY the query string — no quotes, no explanation. "
                "Make it concrete and visual. "
                "For conflict/war topics: use specific location names and event types. "
                "For science topics: use specific process or equipment names. "
                "Example good queries: 'Myanmar refugees border camp 2024', "
                "'pharmacokinetics drug absorption graph', 'quantum computing chip IBM'."
            )),
            HumanMessage(content=(
                f"Research topic: {topic}\n"
                f"Report section heading: {section_title}\n"
                "Generate an image search query for this section."
            )),
        ])
        raw = response.content
        if isinstance(raw, list):
            raw = " ".join(
                p["text"] if isinstance(p, dict) and "text" in p else str(p)
                for p in raw
            )
        query = raw.strip().strip('"').strip("'").split("\n")[0].strip()
        return query if len(query) > 3 else f"{section_title} {topic}"
    except Exception:
        return f"{section_title} {topic}"


# ── Section parser ─────────────────────────────────────────────────────────────

def _extract_sections(report: str) -> list[dict]:
    """
    Extract ## headings that benefit from images.
    Also strips emoji/symbols from title before skip-matching.
    """
    SKIP_TITLES = {
        "executive summary", "key takeaways", "resources",
        "references", "conclusion", "introduction", "overview",
        "research objectives and scope", "core methodologies",
    }
    sections = []
    for m in re.finditer(r'^## (.+)$', report, re.MULTILINE):
        title = m.group(1).strip()
        # Strip emoji and symbols for skip comparison only
        clean = re.sub(r'[^\w\s]', '', title).strip().lower()
        # Also strip leading numbers like "1. " or "1 "
        clean = re.sub(r'^\d+[\.\s]+', '', clean).strip()
        if clean not in SKIP_TITLES:
            sections.append({
                "title":     title,
                "clean":     clean,
                "start_pos": m.end(),
            })
    return sections


# ── Image injection ────────────────────────────────────────────────────────────

def _inject_images(report: str, insertions: list[dict]) -> str:
    """Insert image markdown at the correct positions (descending order)."""
    if not insertions:
        return report
    sorted_ins = sorted(insertions, key=lambda x: x["pos"], reverse=True)
    result = report
    for ins in sorted_ins:
        pos    = ins["pos"]
        img_md = (
            f"\n\n![{ins['alt']}]({ins['url']})\n"
            f"*{ins['alt']} — [Source]({ins['source']})*\n"
        )
        result = result[:pos] + img_md + result[pos:]
    return result


# ── Main node ──────────────────────────────────────────────────────────────────

def image_node(state: ResearchState) -> dict:
    report = state.get("final_report", "")
    query  = state.get("query", "")

    if not report:
        return {"image_insertions": [], "final_report": report}

    sections   = _extract_sections(report)
    insertions = []

    for sec in sections:
        search_query = _generate_image_query(sec["title"], query)
        time.sleep(0.3)

        chosen = _search_image(search_query)
        if not chosen:
            # Try a simpler fallback query using just the section title
            chosen = _search_image(sec["clean"])

        if not chosen:
            continue

        insertions.append({
            "section": sec["title"],
            "query":   search_query,
            "url":     chosen["url"],
            "alt":     chosen["title"],
            "source":  chosen["source"],
            "pos":     sec["start_pos"],
        })

    updated_report = _inject_images(report, insertions)

    return {
        "final_report":     updated_report,
        "image_insertions": insertions,
    }