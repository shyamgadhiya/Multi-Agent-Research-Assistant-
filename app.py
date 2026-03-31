import os
import time
import json
import sqlite3
import tempfile
import datetime
import uuid
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# load_dotenv()

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide",
)

st.markdown("""
<style>
.node-card { border:0.5px solid #e0e0e0; border-left:4px solid #6c63ff; border-radius:10px;
             padding:14px 18px; margin:8px 0; background:#ffffff; font-size:14px; line-height:1.6; }
.node-card.done    { border-left-color:#28a745; background:#f6fff8; }
.node-card.retry   { border-left-color:#dc3545; background:#fff6f6; }
.node-card.running { border-left-color:#f0ad4e; background:#fffdf0; }
.node-header { display:flex; align-items:center; justify-content:space-between; margin-bottom:6px; }
.node-title  { font-size:14px; font-weight:600; color:#1a1a1a; }
.node-status { font-size:12px; color:#666; }
.task-text   { font-size:13px; color:#444; margin-top:4px; padding:6px 10px;
               background:#f0f0f0; border-radius:6px; font-style:italic; }
.task-pill   { display:inline-block; background:#ede9ff; color:#3d35a0; border-radius:20px;
               padding:3px 11px; margin:3px 3px 3px 0; font-size:12px; font-weight:500; }
.gap-pill    { display:inline-block; background:#fdecea; color:#a00; border-radius:20px;
               padding:3px 11px; margin:3px 3px 3px 0; font-size:12px; font-weight:500; }
.source-pill { display:inline-block; background:#e3f6ec; color:#0a6b36; border-radius:20px;
               padding:2px 9px; margin:0 3px; font-size:11px; font-weight:500; }
.score-badge { display:inline-block; padding:2px 12px; border-radius:20px; font-size:13px; font-weight:700; }
.score-pass  { background:#d4edda; color:#155724; }
.score-fail  { background:#f8d7da; color:#721c24; }
.metric-row  { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:16px 0; }
.metric-card { background:#f8f9fa; border:0.5px solid #e0e0e0; border-radius:10px;
               padding:14px 16px; text-align:center; }
.metric-val  { font-size:24px; font-weight:700; color:#1a1a1a; }
.metric-key  { font-size:12px; color:#888; margin-top:2px; }
.file-chip   { display:inline-block; background:#e8f4fd; color:#0c5460; border-radius:6px;
               padding:3px 10px; margin:2px 3px; font-size:12px; }
.collection-box { background:#f8f9fa; border:0.5px solid #dee2e6; border-radius:8px;
                  padding:10px 12px; margin:6px 0; }
.rag-confirm { background:#e8f5e9; border:0.5px solid #a5d6a7; border-radius:8px;
               padding:7px 12px; font-size:12px; color:#1b5e20; }
.web-confirm { background:#e3f2fd; border:0.5px solid #90caf9; border-radius:8px;
               padding:7px 12px; font-size:12px; color:#0d47a1; }
</style>
""", unsafe_allow_html=True)

# ── Keys ───────────────────────────────────────────────────────────────────────
# GOOGLE_API_KEY    = os.environ.get("GOOGLE_API_KEY", "")
# TAVILY_API_KEY    = os.environ.get("TAVILY_API_KEY", "")
# LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY", "")
# LANGCHAIN_PROJECT = os.environ.get("LANGCHAIN_PROJECT", "multi-agent-research-assistant")

GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", "")
LANGCHAIN_API_KEY = st.secrets.get("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = st.secrets.get("LANGCHAIN_PROJECT", "multi-agent-research-assistant")

if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"]    = LANGCHAIN_PROJECT
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

# ── Session state ──────────────────────────────────────────────────────────────
if "active_report"  not in st.session_state: st.session_state.active_report  = None
if "active_query"   not in st.session_state: st.session_state.active_query   = ""
if "thread_id"      not in st.session_state:
    st.session_state.thread_id = f"thread-{uuid.uuid4().hex[:8]}"

HISTORY_DB_PATH = Path(__file__).parent / "research_history.sqlite3"

# ── DB helpers ─────────────────────────────────────────────────────────────────

def _init_history_db():
    with sqlite3.connect(HISTORY_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                report TEXT NOT NULL,
                score REAL NOT NULL,
                collection TEXT NOT NULL,
                thread_id TEXT NOT NULL DEFAULT 'legacy',
                ts TEXT NOT NULL
            )
        """)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(history)").fetchall()}
        if "thread_id" not in cols:
            conn.execute("ALTER TABLE history ADD COLUMN thread_id TEXT NOT NULL DEFAULT 'legacy'")

def _load_history(thread_id: str, limit: int = 50) -> list[dict]:
    with sqlite3.connect(HISTORY_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id,query,report,score,collection,thread_id,ts FROM history "
            "WHERE thread_id=? ORDER BY id DESC LIMIT ?",
            (thread_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]

def _list_threads(limit: int = 30) -> list[dict]:
    with sqlite3.connect(HISTORY_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            WITH latest AS (
                SELECT thread_id, MAX(id) AS last_id, COUNT(*) AS run_count
                FROM history GROUP BY thread_id
            )
            SELECT l.thread_id, l.run_count, h.ts, h.query, l.last_id
            FROM latest l JOIN history h ON h.id = l.last_id
            ORDER BY l.last_id DESC LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]

def _save_to_history(query, report, score, collection, thread_id):
    ts = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
    with sqlite3.connect(HISTORY_DB_PATH) as conn:
        conn.execute(
            "INSERT INTO history (query,report,score,collection,thread_id,ts) VALUES (?,?,?,?,?,?)",
            (query, report, score, collection, thread_id, ts),
        )

def _thread_label(tid: str, meta: dict) -> str:
    m = meta.get(tid)
    if not m: return f"{tid} (new)"
    q = (m.get("query") or "")[:30]
    return f"{tid} | {m.get('run_count',0)} run(s) | {m.get('ts','')} | {q}"

_init_history_db()

# ── Collection / file helpers ──────────────────────────────────────────────────

def _collection_path(name: str) -> str:
    return os.path.join("vectorstores", name.strip().replace(" ", "_").lower())

def _list_collections() -> list[str]:
    base = Path("vectorstores")
    if not base.exists(): return []
    return sorted([d.name for d in base.iterdir() if d.is_dir()])

def _collection_meta_path(collection_name: str) -> Path:
    """JSON file that stores the list of ingested filenames for a collection."""
    return Path(_collection_path(collection_name)) / "meta.json"

def _get_collection_files(collection_name: str) -> list[str]:
    """Return list of filenames ingested into this collection."""
    p = _collection_meta_path(collection_name)
    if not p.exists(): return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []

def _save_collection_meta(collection_name: str, filenames: list[str]):
    p = _collection_meta_path(collection_name)
    p.write_text(json.dumps(filenames))

def _collection_has_index(collection_name: str) -> bool:
    """True only if a real FAISS index exists (not just an empty folder)."""
    base = Path(_collection_path(collection_name))
    return (base / "index.faiss").exists() or (base / "faiss.index").exists()

def _load_all_history(limit: int = 60) -> list[dict]:
    """Load all history items flat — no thread grouping."""
    with sqlite3.connect(HISTORY_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id,query,report,score,collection,thread_id,ts FROM history "
            "ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def _delete_file_from_collection(col_name: str, filename: str):
    """
    Remove a specific file's chunks from the collection by rebuilding
    the FAISS index without that file's documents.
    """
    import shutil, pickle
    from langchain_community.vectorstores import FAISS
    from setup import embeddings

    path    = _collection_path(col_name)
    pkl_path = os.path.join(path, "index.pkl")
    if not os.path.exists(pkl_path):
        return

    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        docstore, id_map = data[0], data[1]
        # Filter out documents from the target file
        keep_docs = [
            doc for doc in docstore._dict.values()
            if doc.metadata.get("source", "") != filename
        ]
        if not keep_docs:
            shutil.rmtree(path, ignore_errors=True)
            return
        # Rebuild index from remaining docs
        texts  = [d.page_content for d in keep_docs]
        metas  = [d.metadata     for d in keep_docs]
        new_vs = FAISS.from_texts(texts, embeddings, metadatas=metas)
        new_vs.save_local(path)
        # Update meta file
        existing_files = _get_collection_files(col_name)
        updated_files  = [f for f in existing_files if f != filename]
        _save_collection_meta(col_name, updated_files)
    except Exception:
        pass


def _ingest(files, collection_name: str):
    """Ingest files into named collection — always overwrites cleanly."""
    import shutil
    from langchain_community.document_loaders import TextLoader, PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from setup import embeddings

    save_path = _collection_path(collection_name)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    all_docs = []
    filenames = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for uf in files:
            p = Path(tmpdir) / uf.name
            p.write_bytes(uf.read())
            loader = PyPDFLoader(str(p)) if uf.name.endswith(".pdf") else TextLoader(str(p))
            docs = loader.load()
            for d in docs:
                d.metadata["source"]     = uf.name
                d.metadata["collection"] = collection_name
            all_docs.extend(docs)
            filenames.append(uf.name)

    if not all_docs:
        raise ValueError("No documents could be loaded from the uploaded files.")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100
    ).split_documents(all_docs)
    FAISS.from_documents(chunks, embeddings).save_local(save_path)

    # Persist file list so the UI can display them
    _save_collection_meta(collection_name, filenames)

# ── PDF export ─────────────────────────────────────────────────────────────────

def _embed_images_in_markdown(markdown_text: str) -> str:
    """
    Replace remote image URLs in markdown with base64 data URIs.
    This makes the .md file fully self-contained — images display
    offline in any markdown viewer that supports HTML img tags.
    Falls back to the original URL if download fails.
    """
    import re, base64
    img_pattern = re.compile(r'(!\[([^\]]*)\]\()([^)]+)(\))')

    def replace_with_base64(match):
        prefix = match.group(1)   # ![alt](
        alt    = match.group(2)
        url    = match.group(3)
        suffix = match.group(4)   # )
        if not url.startswith("http"):
            return match.group(0)
        tmp = _fetch_image_to_tempfile(url)
        if not tmp:
            return match.group(0)   # keep original URL on failure
        try:
            import os
            with open(tmp, "rb") as f:
                data = base64.b64encode(f.read()).decode("ascii")
            ext  = os.path.splitext(tmp)[1].lstrip(".") or "jpeg"
            mime = f"image/{ext}"
            os.unlink(tmp)
            return f"{prefix}data:{mime};base64,{data}{suffix}"
        except Exception:
            return match.group(0)

    return img_pattern.sub(replace_with_base64, markdown_text)


def _fetch_image_to_tempfile(url: str) -> str | None:
    """
    Download a remote image URL to a temp file.
    Returns the temp file path, or None on failure.
    The caller is responsible for deleting the file after use.
    """
    import urllib.request, tempfile, os
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ResearchAssistant/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            ct  = resp.headers.get("Content-Type", "")
            ext = ".jpg"
            if "png"  in ct: ext = ".png"
            elif "gif" in ct: ext = ".gif"
            elif "webp" in ct: ext = ".webp"
            data = resp.read()
        # fpdf2 can handle jpg/png/gif — skip webp (not supported)
        if ext == ".webp":
            return None
        fd, path = tempfile.mkstemp(suffix=ext)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return path
    except Exception:
        return None


def _export_pdf(markdown_text: str):
    """
    Convert markdown report to PDF using fpdf2 with DejaVu Unicode font.
    DejaVu is bundled with fpdf2 - no external font files needed.
    Supports full Unicode including bullet, arrows, and all special characters.
    Returns (pdf_bytes, None) on success or (None, error_string) on failure.
    """
    import re, os, tempfile
    try:
        from fpdf import FPDF

        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.set_margins(left=22, top=22, right=22)
        pdf.set_auto_page_break(auto=True, margin=22)

        # Find a Unicode-capable font on this system.
        # Priority: DejaVu in project folder → Arial on Windows → fallback to latin-1 Helvetica
        import os as _os

        def _find_font(filenames):
            """Search project folder and common Windows/Linux font dirs."""
            search_dirs = [
                _os.path.dirname(_os.path.abspath(__file__)),  # project folder
                ".",                                            # cwd
                r"C:\Windows\Fonts",                         # Windows
                "/usr/share/fonts/truetype/dejavu",            # Ubuntu/Debian
                "/usr/share/fonts/TTF",                        # Arch
                "/Library/Fonts",                              # macOS
            ]
            for d in search_dirs:
                for fname in filenames:
                    p = _os.path.join(d, fname)
                    if _os.path.exists(p):
                        return p
            return None

        # Try DejaVu (best Unicode coverage, can be placed in project folder)
        _dejavu     = _find_font(["DejaVuSans.ttf",         "DejaVuSansCondensed.ttf"])
        _dejavu_b   = _find_font(["DejaVuSans-Bold.ttf",    "DejaVuSansCondensed-Bold.ttf"])
        _dejavu_i   = _find_font(["DejaVuSans-Oblique.ttf", "DejaVuSansCondensed-Oblique.ttf"])
        _dejavu_bi  = _find_font(["DejaVuSans-BoldOblique.ttf", "DejaVuSansCondensed-BoldOblique.ttf"])

        # Try Arial which ships with every Windows install and has good Unicode coverage
        _arial      = _find_font(["arial.ttf",    "Arial.ttf"])
        _arial_b    = _find_font(["arialbd.ttf",  "Arial Bold.ttf",  "ArialBold.ttf"])
        _arial_i    = _find_font(["ariali.ttf",   "Arial Italic.ttf"])
        _arial_bi   = _find_font(["arialbi.ttf",  "Arial Bold Italic.ttf"])

        if _dejavu:
            pdf.add_font("UniFont",      fname=_dejavu)
            pdf.add_font("UniFont", style="B",  fname=_dejavu_b  or _dejavu)
            pdf.add_font("UniFont", style="I",  fname=_dejavu_i  or _dejavu)
            pdf.add_font("UniFont", style="BI", fname=_dejavu_bi or _dejavu)
            FONT = "UniFont"
        elif _arial:
            pdf.add_font("UniFont",      fname=_arial)
            pdf.add_font("UniFont", style="B",  fname=_arial_b  or _arial)
            pdf.add_font("UniFont", style="I",  fname=_arial_i  or _arial)
            pdf.add_font("UniFont", style="BI", fname=_arial_bi or _arial)
            FONT = "UniFont"
        else:
            # Last resort: Helvetica (latin-1 only — bullets replaced with "-")
            FONT = "Helvetica"

        pdf.add_page()
        usable_w   = pdf.w - pdf.l_margin - pdf.r_margin

        def _u(text: str) -> str:
            """If using Helvetica fallback, replace non-latin-1 chars cleanly."""
            if FONT == "Helvetica":
                return (text
                    .replace("•", "-")   # bullet
                    .replace("–", "-")   # en-dash
                    .replace("—", "--")  # em-dash
                    .replace("‘", "'").replace("’", "'")
                    .replace("“", '"').replace("”", '"')
                    .encode("latin-1", errors="replace").decode("latin-1"))
            return text
        temp_files = []

        def write_inline(text, size=11, color=(30, 30, 30)):
            if not text.strip():
                return
            # Match: **bold**, [text](url), [https://bare-url], plain https://url
            token_pat = re.compile(
                r"(\*\*(.+?)\*\*"             # group 1/2: **bold**
                r"|\[([^\]]+)\]\(([^)]+)\)"   # group 3/4: [text](url)
                r"|\[(https?://[^\]]+)\]"      # group 5:   [https://bare-url]
                r"|(https?://[^\s\)\],;]+)"    # group 6:   plain https://url
                r")"
            )
            last = 0
            pdf.set_x(pdf.l_margin)
            for m in token_pat.finditer(text):
                before = text[last:m.start()]
                if before:
                    pdf.set_font(FONT, "", size)
                    pdf.set_text_color(*color)
                    pdf.write(6, _u(before))

                g = m.group(0)
                if g.startswith("**"):
                    # **bold**
                    pdf.set_font(FONT, "B", size)
                    pdf.set_text_color(*color)
                    pdf.write(6, _u(m.group(2)))
                    pdf.set_font(FONT, "", size)
                elif m.group(3) and m.group(4):
                    # [text](url) — proper markdown link
                    pdf.set_font(FONT, "U", size)
                    pdf.set_text_color(0, 82, 204)
                    pdf.write(6, _u(m.group(3)), link=m.group(4))
                    pdf.set_font(FONT, "", size)
                    pdf.set_text_color(*color)
                elif m.group(5):
                    # [https://bare-url] — show domain as link text
                    url = m.group(5)
                    try:
                        from urllib.parse import urlparse as _up
                        label = _up(url).netloc.replace("www.", "") or url[:40]
                    except Exception:
                        label = url[:40]
                    pdf.set_font(FONT, "U", size)
                    pdf.set_text_color(0, 82, 204)
                    pdf.write(6, _u(label), link=url)
                    pdf.set_font(FONT, "", size)
                    pdf.set_text_color(*color)
                elif m.group(6):
                    # plain https://url
                    url = m.group(6)
                    try:
                        from urllib.parse import urlparse as _up
                        label = _up(url).netloc.replace("www.", "") or url[:40]
                    except Exception:
                        label = url[:40]
                    pdf.set_font(FONT, "U", size)
                    pdf.set_text_color(0, 82, 204)
                    pdf.write(6, _u(label), link=url)
                    pdf.set_font(FONT, "", size)
                    pdf.set_text_color(*color)

                last = m.end()
            tail = text[last:]
            if tail:
                pdf.set_font(FONT, "", size)
                pdf.set_text_color(*color)
                pdf.write(6, _u(tail))
            pdf.ln()

        def cell(text, style="", size=11, h=6, color=(30, 30, 30)):
            if not text.strip():
                return
            pdf.set_font(FONT, style, size)
            pdf.set_text_color(*color)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(usable_w, h, _u(text))

        def embed_image(url, alt):
            tmp = _fetch_image_to_tempfile(url)
            if not tmp:
                return
            temp_files.append(tmp)
            try:
                pdf.set_x(pdf.l_margin)
                pdf.image(tmp, x=pdf.l_margin, w=usable_w * 0.80)
                if alt:
                    pdf.set_font(FONT, "I", 9)
                    pdf.set_text_color(120, 120, 120)
                    pdf.set_x(pdf.l_margin)
                    pdf.multi_cell(usable_w, 5, _u(alt[:120]))
                pdf.ln(3)
            except Exception:
                pass

        img_pat     = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)$")
        caption_pat = re.compile(r"^\*(.+?)\*$")

        lines = markdown_text.split("\n")
        i = 0
        while i < len(lines):
            s = lines[i].strip()

            img_m = img_pat.match(s)
            if img_m:
                if i + 1 < len(lines) and caption_pat.match(lines[i + 1].strip()):
                    i += 1
                embed_image(img_m.group(2), img_m.group(1))
                i += 1
                continue

            if s.startswith("# "):
                cell(s[2:],  style="B", size=18, h=10, color=(40, 40, 40))
                pdf.ln(2)
            elif s.startswith("## "):
                cell(s[3:],  style="B", size=14, h=9,  color=(61, 53, 160))
                pdf.ln(1)
            elif s.startswith("### "):
                cell(s[4:],  style="B", size=12, h=8,  color=(60, 60, 60))
            elif s.startswith("- ") or s.startswith("* "):
                pdf.set_x(pdf.l_margin)
                pdf.set_font(FONT, "", 11)
                pdf.set_text_color(30, 30, 30)
                pdf.write(6, _u("  • "))
                write_inline(s[2:])
            elif re.match(r"^\d+\. ", s):
                write_inline(s)
            elif s in ("", "---", "***"):
                pdf.ln(3)
            else:
                write_inline(s)

            i += 1

        pdf_bytes = bytes(pdf.output())
        for tmp in temp_files:
            try:
                os.unlink(tmp)
            except Exception:
                pass
        return pdf_bytes, None

    except ImportError:
        return None, "fpdf2 not installed - run: pip install fpdf2"
    except Exception as e:
        return None, str(e)

def _pills(items, css="task-pill"):
    return "".join(f'<span class="{css}">{t[:65]}{"…" if len(t)>65 else ""}</span>' for t in items)

def _score_html(score, threshold):
    cls = "score-pass" if score >= threshold else "score-fail"
    return f'<span class="score-badge {cls}">{score:.2f}</span>'

def _check_keys():
    errors = []
    if not GOOGLE_API_KEY: errors.append("GOOGLE_API_KEY missing in .env")
    if not TAVILY_API_KEY: errors.append("TAVILY_API_KEY missing in .env")
    return errors

# ── Resource renderer ─────────────────────────────────────────────────────────

def _render_resources(resources_raw: str, findings: list[dict]):
    """Render the Resources section as styled cards with links."""
    st.markdown('<div class="resources-section">', unsafe_allow_html=True)
    st.markdown('<div class="resources-title">📚 Resources</div>', unsafe_allow_html=True)

    # Collect structured sources from findings for richer metadata
    rag_sources = {}
    web_sources = {}
    for f in findings:
        for rs in f.get("rag_sources", []):
            key = (rs.get("file", ""), rs.get("collection", ""))
            if key not in rag_sources:
                rag_sources[key] = rs
        for ws in f.get("web_sources", []):
            url = ws.get("url", "").strip()
            if url and url not in web_sources:
                web_sources[url] = ws

    # RAG documents
    if rag_sources:
        st.markdown('<div class="resources-group"><div class="resources-group-label">Documents (RAG)</div>', unsafe_allow_html=True)
        for (fname, col), _ in rag_sources.items():
            col_tag = f' &nbsp;<span class="res-tag">{col}</span>' if col and col != "none" else ""
            st.markdown(
                f'<div class="res-card"><span class="res-icon">📄</span><div class="res-body"><div class="res-title">{fname}{col_tag}</div><div class="res-url">Local document · collection: {col or "—"}</div></div></div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # Web sources
    if web_sources:
        st.markdown('<div class="resources-group"><div class="resources-group-label">Web Sources</div>', unsafe_allow_html=True)
        for url, ws in web_sources.items():
            title = ws.get("title", "") or url
            # Truncate long titles/urls for display
            display_title = title[:90] + "…" if len(title) > 90 else title
            display_url   = url[:80]   + "…" if len(url)   > 80 else url
            st.markdown(
                f'<a href="{url}" target="_blank" style="text-decoration:none"><div class="res-card"><span class="res-icon">🌐</span><div class="res-body"><div class="res-title">{display_title}</div><div class="res-url">{display_url}</div></div></div></a>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # Fallback: if findings had no structured sources, parse the markdown text
    if not rag_sources and not web_sources and resources_raw.strip():
        st.markdown(resources_raw)

    st.markdown('</div>', unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Model settings are fixed — not shown to user
    critic_threshold = 0.85
    max_retries      = 5

    # ── Document Collections (top of sidebar) ─────────────────────────────────
    st.markdown("**📂 Document Collections**")

    with st.expander("➕ Add / update a collection", expanded=False):
        new_col_name   = st.text_input("Collection name", placeholder="e.g. Pharma Research", key="new_col_input")
        uploaded_files = st.file_uploader(
            "Upload files (.pdf, .txt, .md)",
            accept_multiple_files=True,
            type=["pdf", "txt", "md"],
            key="uploader",
        )
        if uploaded_files and new_col_name.strip():
            if st.button(f"🔄 Ingest into \'{new_col_name}\'", use_container_width=True, type="primary", key="ingest_btn"):
                with st.spinner(f"Embedding and saving to \'{new_col_name}\'…"):
                    _ingest(uploaded_files, new_col_name.strip())
                st.success(f"✅ \'{new_col_name}\' ready!")
                st.rerun()
        elif uploaded_files and not new_col_name.strip():
            st.caption("⚠️ Enter a collection name first.")

    collections        = _list_collections()
    active_collection  = None
    active_file_filter = None

    if not collections:
        st.caption("No collections yet. Add one above.")
    else:
        active_collection = st.selectbox(
            "Active collection",
            options=collections,
            key="active_col_select",
        )
        col_files = _get_collection_files(active_collection) if active_collection else []
        has_index = _collection_has_index(active_collection) if active_collection else False

        if active_collection:
            if st.button(f"🗑 Delete collection \'{active_collection}\'", use_container_width=True, key="del_col"):
                import shutil
                shutil.rmtree(_collection_path(active_collection), ignore_errors=True)
                st.success(f"Deleted \'{active_collection}\'")
                st.rerun()

            if not has_index:
                st.warning("No ingested index found. Upload and ingest files.")
            elif col_files:
                file_options   = ["All files"] + col_files
                selected_file  = st.selectbox("File", options=file_options, key="file_select",
                                               help="Limit RAG to a specific file")
                active_file_filter = None if selected_file == "All files" else selected_file
                if active_file_filter:
                    if st.button(f"🗑 Remove \'{active_file_filter}\'", key="del_file"):
                        _delete_file_from_collection(active_collection, active_file_filter)
                        st.success(f"Removed \'{active_file_filter}\'")
                        st.rerun()
                chips = "".join(f'<span class="file-chip">📄 {f}</span>' for f in col_files)
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.caption("No file metadata found. Re-ingest to enable file filtering.")

    st.divider()

    # ── Flat chat history ──────────────────────────────────────────────────────
    st.markdown("**🕘 History**")
    # st.caption(f"Thread: `{st.session_state.thread_id[:14]}…`")
    if st.button("➕ New Chat", use_container_width=True, key="new_chat_btn"):
        st.session_state.thread_id    = f"thread-{uuid.uuid4().hex[:8]}"
        st.session_state.active_report = None
        st.session_state.active_query  = ""
        st.rerun()

    all_history = _load_all_history(limit=60)
    if not all_history:
        st.caption("No runs yet.")
    else:
        for h in all_history:
            q_short = h["query"][:50] + "…" if len(h["query"]) > 50 else h["query"]
            label   = f"{q_short}\n\n_{h['ts']} · {h['score']:.2f}_"
            if st.button(label, key=f"hist_{h['id']}", use_container_width=True):
                st.session_state.active_report = h["report"]
                st.session_state.active_query  = h["query"]
                st.rerun()

    st.divider()




# ── Main page ──────────────────────────────────────────────────────────────────
st.markdown("# 🔬 Multi-Agent Research Assistant")
st.markdown(
    "Decomposes your question → parallel **RAG + Web** research → "
    "self-critique loop → cited report."
)

key_errors = _check_keys()
if key_errors:
    for e in key_errors: st.error(f"⚠️ {e}")
    st.info("Add the missing keys to your `.env` file and restart.")
    st.stop()

st.divider()

# Show saved report if user clicked history
if st.session_state.active_report:
    st.info(f"📋 Showing saved report for: **{st.session_state.active_query}**")
    if st.button("✖ Clear and start new research"):
        st.session_state.active_report = None
        st.session_state.active_query  = ""
        st.rerun()
    with st.container(border=True):
        st.markdown(st.session_state.active_report)
    col_md, col_pdf = st.columns(2)
    with col_md:
        st.download_button("⬇️ Download .md", data=st.session_state.active_report,
            file_name="report.md", mime="text/markdown", use_container_width=True)
    with col_pdf:
        pdf_bytes, pdf_err = _export_pdf(st.session_state.active_report)
        if pdf_bytes:
            st.download_button("⬇️ Download PDF", data=pdf_bytes,
                file_name="report.pdf", mime="application/pdf", use_container_width=True)
        else:
            st.error(f"PDF failed: {pdf_err}")
    st.stop()

# ── Query input ────────────────────────────────────────────────────────────────
query = st.text_area(
    "**Research question**",
    placeholder="e.g. What are the key findings of the uploaded document?",
    height=100,
)

# ── Mode row: WEB | RAG | 🖼 Images toggle | Run button ────────────────────────
mcol1, mcol2, mcol3, mcol4 = st.columns([1.1, 1.3, 1.8, 1.2])

with mcol1:
    mode = st.radio(
        "mode",
        options=["🌐 Web", "📄 RAG"],
        horizontal=True,
        label_visibility="collapsed",
        key="research_mode_radio",
    )
    use_rag = mode == "📄 RAG"

with mcol2:
    images_enabled = st.toggle(
        "🖼️ Images",
        value=False,
        help="Embed relevant images into each report section (adds ~20s per section).",
        key="images_toggle",
    )

with mcol3:
    # Confirmation / status banner
    if use_rag:
        if not active_collection:
            st.markdown(
                '<div style="background:#fff3e0;border:0.5px solid #ffcc02;border-radius:8px;'
                'padding:6px 12px;font-size:12px;color:#e65100;margin-top:4px">'
                '⚠️ No collection — add one in sidebar</div>',
                unsafe_allow_html=True,
            )
        elif not _collection_has_index(active_collection):
            st.markdown(
                '<div style="background:#fff3e0;border:0.5px solid #ffcc02;border-radius:8px;'
                'padding:6px 12px;font-size:12px;color:#e65100;margin-top:4px">'
                f'⚠️ Collection has no index — ingest files</div>',
                unsafe_allow_html=True,
            )
        else:
            file_label = f"📄 {active_file_filter}" if active_file_filter else f"📂 {active_collection} — all files"
            st.markdown(
                f'<div class="rag-confirm" style="margin-top:4px">✅ RAG · {file_label}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="web-confirm" style="margin-top:4px">🌐 Web search</div>',
            unsafe_allow_html=True,
        )

with mcol4:
    run_btn = st.button("🚀 Run Research", type="primary", use_container_width=True)

# Grey-out collection display in sidebar when WEB mode is active
if not use_rag:
    st.markdown(
        """<style>
        div[data-testid="stSelectbox"] label,
        div[data-testid="stSelectbox"] > div:first-child {
            opacity: 0.4;
            pointer-events: none;
        }
        .file-chip { opacity: 0.4 !important; }
        </style>""",
        unsafe_allow_html=True,
    )

st.divider()

# Compute effective vectorstore path
effective_collection = active_collection if use_rag else None
effective_vectorstore_path = _collection_path(effective_collection) if effective_collection else ""

# ── Validation before run ──────────────────────────────────────────────────────
if run_btn:
    errors = []
    if not query.strip():
        errors.append("Please enter a research question.")

    if use_rag:
        if not active_collection:
            errors.append("RAG mode requires a collection. Add one in the sidebar.")
        elif not _collection_has_index(active_collection):
            errors.append(
                f"Collection '{active_collection}' has no ingested index. "
                "Open the sidebar, upload files, and click Ingest before running."
            )

    if errors:
        for e in errors:
            st.error(f"⚠️ {e}")
        st.stop()

    # ── Pipeline ───────────────────────────────────────────────────────────────
    import critic_node as _cn
    _cn.PASS_THRESHOLD = critic_threshold
    _cn.MAX_RETRIES    = max_retries

    from graph import app

    # If user selected a specific file, embed that as a filter hint in the query
    query_with_filter = query.strip()
    if use_rag and active_file_filter:
        query_with_filter = f"[Search only in file: {active_file_filter}] {query.strip()}"

    initial_state = {
        "query":            query_with_filter,
        "sub_tasks":        [],
        "findings":         [],
        "critic_score":     0.0,
        "gaps":             [],
        "retry_count":      0,
        "final_report":     "",
        "vectorstore_path": effective_vectorstore_path,
        "active_file":      active_file_filter or "",
        "images_enabled":   images_enabled,
        "image_insertions": [],
        "_topic_difficulty": "general",
        "_research_mode":    "Web only",
    }

    st.markdown("### ⚡ Pipeline Progress")
    progress_bar = st.progress(0, text="Starting pipeline…")

    WEIGHTS = {"planner_node": 10, "researcher_node": 55, "critic_node": 20, "writer_node": 15}
    progress_so_far  = 0
    researcher_count = 0
    researcher_total = 0
    final_state      = dict(initial_state)
    final_state["findings"] = []

    for step in app.stream(initial_state, stream_mode="updates"):
        node_name   = list(step.keys())[0]
        node_output = step[node_name]

        for k, v in node_output.items():
            if k == "findings" and isinstance(v, list):
                final_state["findings"] = final_state.get("findings", []) + v
            else:
                final_state[k] = v

        if node_name == "planner_node":
            tasks            = node_output.get("sub_tasks", [])
            researcher_total = len(tasks)
            retry            = node_output.get("retry_count", 0)
            is_retry         = retry > 0
            label = "🔁 Replanning" if is_retry else "🧠 Planner Agent"
            css   = "retry" if is_retry else "done"
            st.markdown(
                f'<div class="node-card {css}">'
                f'<div class="node-header"><span class="node-title">{label}</span>'
                f'<span class="node-status">{"Retry #"+str(retry) if is_retry else str(len(tasks))+" sub-tasks"}</span></div>'
                f'{_pills(tasks)}</div>',
                unsafe_allow_html=True,
            )
            progress_so_far = min(progress_so_far + WEIGHTS["planner_node"], 95)
            progress_bar.progress(int(progress_so_far), text=f"Planner done · {len(tasks)} tasks dispatched…")

        elif node_name == "researcher_node":
            researcher_count += 1
            all_findings = final_state.get("findings", [])
            f0        = all_findings[-1] if all_findings else {}
            task_text = f0.get("task", "")
            rag_used  = f0.get("rag_used",  False)
            web_used  = f0.get("web_used",  False)
            web_skip  = f0.get("web_skipped", False)
            crag      = f0.get("crag", {})

            # Source badges
            source_badges = (
                ('<span class="source-pill">RAG</span>' if rag_used else "") +
                ('<span class="source-pill">Web</span>' if web_used else "") +
                ('<span class="source-pill" style="background:#fff3e0;color:#e65100">Web skipped</span>' if web_skip else "")
            )

            # CRAG summary line
            crag_html = ""
            if crag:
                verdict   = crag.get("chunk_verdict", "")
                kept      = crag.get("chunks_kept",  0)
                total     = crag.get("chunks_total", 0)
                g_score   = crag.get("grounding_score", 1.0)
                low_conf  = crag.get("low_confidence", False) and g_score < 0.40

                if verdict == "not_used":
                    # Web-only mode — no RAG chunks to show
                    crag_html = (
                        f'<div style="margin-top:6px;font-size:11px;color:#666;display:flex;gap:12px;flex-wrap:wrap">'
                        f'<span>Web mode</span>'
                        f'<span>Grounding: <b style="color:#28a745">{g_score:.0%}</b></span>'
                        f'</div>'
                    )
                else:
                    v_color   = "#155724" if verdict == "relevant" else ("#e65100" if verdict == "irrelevant" else "#7b5b00")
                    conf_color = "#dc3545" if low_conf else "#28a745"
                    crag_html = (
                        f'<div style="margin-top:6px;font-size:11px;color:#666;display:flex;gap:12px;flex-wrap:wrap">'
                        f'<span>Chunks: <b style="color:{v_color}">{kept}/{total} kept ({verdict})</b></span>'
                        f'<span>Grounding: <b style="color:{conf_color}">{g_score:.0%}</b></span>'
                        f'{"<span style=\"color:#dc3545;font-weight:500\">⚠ Low confidence</span>" if low_conf else ""}'
                        f'</div>'
                    )

            count_label = f"{researcher_count}/{researcher_total}" if researcher_total else str(researcher_count)
            st.markdown(
                f'<div class="node-card done">'
                f'<div class="node-header"><span class="node-title">🔍 Researcher {count_label}</span>'
                f'<span class="node-status">{source_badges}</span></div>'
                f'<div class="task-text">{task_text[:150]}{"…" if len(task_text)>150 else ""}</div>'
                f'{crag_html}'
                f'</div>',
                unsafe_allow_html=True,
            )
            per = WEIGHTS["researcher_node"] / max(researcher_total, 1)
            progress_so_far = min(progress_so_far + per, 95)
            progress_bar.progress(int(progress_so_far), text=f"Researcher {count_label} done…")

        elif node_name == "critic_node":
            score      = node_output.get("critic_score", 0.0)
            gaps       = node_output.get("gaps", [])
            difficulty = node_output.get("_topic_difficulty", "general")
            mode       = node_output.get("_research_mode", "")

            # Compute the effective threshold the critic actually used
            retry_so_far = final_state.get("retry_count", 0)
            eff_threshold = 0.60 if difficulty == "specialist" else critic_threshold
            eff_threshold = max(eff_threshold - min(retry_so_far * 0.05, 0.10), 0.50)

            passed  = score >= eff_threshold
            css     = "done" if passed else "retry"
            verdict = "✅ Quality passed — proceeding to writer" if passed else "⚠️ Quality insufficient — retrying planner"

            # Meta line: shows effective threshold + topic type + mode
            meta_html = (
                f'<div style="margin-top:6px;font-size:11px;color:#666;display:flex;gap:14px;flex-wrap:wrap">'
                f'<span>Threshold: <b>{eff_threshold:.2f}</b></span>'
                f'<span>Topic: <b style="color:{"#3d35a0" if difficulty=="specialist" else "#444"}">{difficulty}</b></span>'
                f'<span>Mode: <b>{mode}</b></span>'
                f'</div>'
            )
            gaps_html = (
                f'<div style="margin-top:8px">Gaps: {_pills(gaps, "gap-pill")}</div>'
                if gaps else ""
            )
            st.markdown(
                f'<div class="node-card {css}">'
                f'<div class="node-header"><span class="node-title">🧐 Critic Agent</span>'
                f'<span class="node-status">{_score_html(score, eff_threshold)}</span></div>'
                f'{verdict}'
                f'{meta_html}'
                f'{gaps_html}</div>',
                unsafe_allow_html=True,
            )
            progress_so_far = min(progress_so_far + WEIGHTS["critic_node"], 95)
            progress_bar.progress(int(progress_so_far), text="Critic evaluation done…")

        elif node_name == "writer_node":
            st.markdown(
                '<div class="node-card done">'
                '<div class="node-header"><span class="node-title">✍️ Writer Agent</span>'
                '<span class="node-status">Synthesising report…</span></div>'
                'Combining all findings into a structured cited report.'
                '</div>',
                unsafe_allow_html=True,
            )
            pct = 95 if images_enabled else 100
            progress_bar.progress(pct, text="Report written…" if images_enabled else "Research complete!")

        elif node_name == "image_node":
            insertions = node_output.get("image_insertions", [])
            count      = len(insertions)
            st.markdown(
                f'<div class="node-card done">'
                f'<div class="node-header">'
                f'<span class="node-title">🖼️ Image Agent</span>'
                f'<span class="node-status">{count} image(s) added</span></div>'
                f'Found and embedded {count} relevant image(s) across report sections.</div>',
                unsafe_allow_html=True,
            )
            progress_bar.progress(100, text="Research complete!")

    # ── Report ─────────────────────────────────────────────────────────────────
    report = final_state.get("final_report", "")

    if report:
        final_score = final_state.get("critic_score", 0.0)

        _save_to_history(
            query.strip(), report, final_score,
            effective_collection or "web only",
            st.session_state.thread_id,
        )

        st.divider()
        st.markdown("### 📄 Final Report")

        st.markdown(
            f'<div class="metric-row">'
            f'<div class="metric-card"><div class="metric-val">{final_score:.2f}</div><div class="metric-key">Critic Score</div></div>'
            f'<div class="metric-card"><div class="metric-val">{final_state.get("retry_count",0)}</div><div class="metric-key">Retries</div></div>'
            f'<div class="metric-card"><div class="metric-val">{len(final_state.get("sub_tasks",[]))}</div><div class="metric-key">Sub-tasks</div></div>'
            f'<div class="metric-card"><div class="metric-val">{researcher_count}</div><div class="metric-key">Researchers</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Split report body from Resources section — robust regex handles
        # emoji, spacing variations, and missing newline before ---
        import re as _re
        _res_match = _re.search(r'\n?---\n## [^\n]*Resources[^\n]*\n', report)
        if _res_match:
            report_body  = report[:_res_match.start()]
            resources_raw = report[_res_match.start():]
        else:
            report_body, resources_raw = report, ""

        with st.container(border=True):
            st.markdown(report_body)

        # Render Resources section as styled cards
        if resources_raw:
            _render_resources(resources_raw, final_state.get("findings", []))

        col_md, col_pdf, col_ls = st.columns([1, 1, 2])
        with col_md:
            md_with_images = _embed_images_in_markdown(report)
            st.download_button("⬇️ Download .md", data=md_with_images,
                file_name="research_report.md", mime="text/markdown", use_container_width=True)
        with col_pdf:
            pdf_bytes, pdf_err = _export_pdf(report)
            if pdf_bytes:
                st.download_button("⬇️ Download PDF", data=pdf_bytes,
                    file_name="research_report.pdf", mime="application/pdf", use_container_width=True)
            else:
                st.error(f"PDF failed: {pdf_err}")
    else:
        st.warning("Pipeline finished but no report was generated. Check your .env and try again.")
